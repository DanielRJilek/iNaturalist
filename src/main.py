import torch
import torchvision
from YOLOV1 import YOLOV1
from dataset import build_loaders, load_dataset_stats, load_class_indices, create_mapping
from subset import prompt_order
from train import run_training

def main():
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn', force=True)
    batch_size_train = 64
    batch_size_test = 128
    epochs = 1
    random_seed = 1
    learning_rate = 1e-4  
    momentum = 0.9  
    weight_decay=5e-4
    test_interval = 1
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True

    imageNet_mean = [0.485, 0.456, 0.406]
    imageNet_std = [0.229, 0.224, 0.225]

    while True:
        choice = input("Start a new run or continue? [n/c]: ").strip().lower()
        if choice in ("n", "c"):
            break
        print("Enter n or c.")
    while True:
        model_type = input("Choose model type [y/18/50/tiny]: ").strip().lower()
        if model_type in ("y", "18", "50", "tiny"):
            break
        print("Enter y for YOLOV1 or 18 for ResNet18 or 50 for ResNet50 or tiny for TinyNet.")
    order = prompt_order()

    # Get indices for both train and valid sets
    indices_train = load_class_indices(8, order, train=True)
    indices_valid = load_class_indices(8, order, train=False)
    map_target = create_mapping(order)
    num_classes = len(map_target.mapping_dict)
    print(f"Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "y":
        filename = f"models/checkpoint_yolo_{order}.pt"
        model = YOLOV1(num_classes=186, initial_kernel_size=7).to(device)
        my_dataset_mean, my_dataset_std = load_dataset_stats()
        train_loader, test_loader = build_loaders(my_dataset_mean, my_dataset_std, indices_train, indices_valid, map_target, batch_size_train, batch_size_test)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif model_type == "18":
        filename = f"models/checkpoint_resnet18_{order}.pt"
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.to(device)
        train_loader, test_loader = build_loaders(imageNet_mean, imageNet_std, indices_train, indices_valid, map_target, batch_size_train, batch_size_test)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif model_type == "50":
        filename = f"models/checkpoint_resnet50_{order}.pt"
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.to(device)
        train_loader, test_loader = build_loaders(imageNet_mean, imageNet_std, indices_train, indices_valid, map_target, batch_size_train, batch_size_test)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif model_type == "tiny":
        filename = f"models/checkpoint_convnext_tiny_{order}.pt"
        model = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        model.to(device)
        model = model.to(memory_format=torch.channels_last)
        train_loader, test_loader = build_loaders(imageNet_mean, imageNet_std, indices_train, indices_valid, map_target, batch_size_train, batch_size_test)
        weight_decay = 0.05
        learning_rate = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if choice == "c":
        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        train_losses = checkpoint["train_losses"].tolist()
        test_losses = checkpoint["test_losses"].tolist()
        top_1_accuracy = checkpoint["top_1_accuracy"].tolist()
        top_5_accuracy = checkpoint["top_5_accuracy"].tolist()
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    else:
        start_epoch = 0
        train_losses, test_losses, top_1_accuracy, top_5_accuracy = [], [], [], []
    
    new_train,new_test,new_top1,new_top5 = run_training(model, train_loader, test_loader, optimizer, epochs, device, test_interval, scheduler=None, start_epoch=start_epoch)
    train_losses.extend(new_train)
    test_losses.extend(new_test)
    top_1_accuracy.extend(new_top1)
    top_5_accuracy.extend(new_top5)
    torch.save({
        "model": model.state_dict(),
        "epoch": start_epoch + epochs,
        "train_losses": torch.tensor(train_losses),
        "test_losses": torch.tensor(test_losses),
        "top_1_accuracy": torch.tensor(top_1_accuracy),
        "top_5_accuracy": torch.tensor(top_5_accuracy),
        "optimizer": optimizer.state_dict(),
    }, filename)

if __name__ == "__main__":
    main()