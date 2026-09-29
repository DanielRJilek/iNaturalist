import torch
from YOLOV1 import YOLOV1
from dataset import build_loaders, load_dataset_stats, load_class_indices, create_mapping
from train import run_training

def main():
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn', force=True)
    batch_size_train = 64
    batch_size_test = 500
    epochs = 5
    random_seed = 1
    learning_rate = 1e-3    
    momentum = 0.9  
    weight_decay=5e-4
    test_interval = 1
    patience = 2
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True

    while True:
        choice = input("Start a new run or continue? [n/c]: ").strip().lower()
        if choice in ("n", "c"):
            break
        print("Enter n or c.")

    my_dataset_mean, my_dataset_std = load_dataset_stats()
    # Get mammal indices for both train and valid sets
    mammal_indices_train = load_class_indices(8, "data/stats/mammal_indices_train.json")
    mammal_indices_valid = load_class_indices(8, "data/stats/mammal_indices_valid.json")
    map_target = create_mapping()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = YOLOV1(num_classes=186, initial_kernel_size=7).to(device)
    train_loader, test_loader = build_loaders(my_dataset_mean, my_dataset_std, mammal_indices_train, mammal_indices_valid, map_target, batch_size_train, batch_size_test)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)

    if choice == "c":
        checkpoint = torch.load("models/checkpoint.pt", map_location="cpu", weights_only=True)
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
    }, "models/checkpoint.pt")

if __name__ == "__main__":
    main()