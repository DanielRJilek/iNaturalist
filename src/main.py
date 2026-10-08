# main.py
# Daniel Jilek, 2026
# This file contains the main harness for training and validating a model on the training data and validating on the test data.
# It prompts the user to choose a model type and order, and then trains and validates the model.

import json
import torch
import torchvision
from YOLOV1 import YOLOV1
from dataset import build_loaders, load_dataset_stats, load_class_indices, create_mapping
from subset import prompt_order
from train import run_training

# Main function for training and validating a model on the training data and validating on the test data.
def main():
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn', force=True)
    random_seed = 1
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True

    # Prompt the user to start a new run or continue with a previous run, and choose a model type and order.
    while True:
        choice = input("Start a new run or continue? [n/c]: ").strip().lower()
        if choice in ("n", "c"):
            break
        print("Enter n or c.")
    while True:
        model_type = input("Choose model type [y/18/50/tiny/small]: ").strip().lower()
        if model_type in ("y", "18", "50", "tiny", "small"):
            break
        print("Enter y for YOLOV1 or 18 for ResNet18 or 50 for ResNet50 or tiny for TinyNet or small for SmallNet.")
    order = prompt_order()

    # Get indices for both train and valid sets
    indices_train = load_class_indices(8, order, train=True)
    indices_valid = load_class_indices(8, order, train=False)
    map_target = create_mapping(order)
    num_classes = len(map_target.mapping_dict)
    print(f"Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("config/train.json", "r") as f:
        cfg = json.load(f)
    epochs = cfg["epochs"]
    test_interval = cfg["test_interval"]
    image_size = cfg["image_size"]
    model_cfg = cfg["models"][model_type]
    learning_rate = model_cfg["learning_rate"]
    weight_decay = model_cfg["weight_decay"]
    batch_size_train = model_cfg["batch_size_train"]
    batch_size_test = model_cfg["batch_size_test"]
    filename = f"models/checkpoint_{model_cfg['name']}_{order}.pt"
    if model_cfg["normalize"] == "imagenet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = load_dataset_stats()
    train_loader, test_loader = build_loaders(
        mean, std, indices_train, indices_valid, map_target,
        batch_size_train, batch_size_test
    )
    
    # Load the model and build the loaders for the training and validation data.
    # Set final layer to the number of classes.
    if model_type == "y":
        model = YOLOV1(num_classes=186, initial_kernel_size=7).to(device)
    elif model_type == "18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.to(device)
    elif model_type == "50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.to(device)
    elif model_type == "tiny":
        model = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        model.to(device)
        model = model.to(memory_format=torch.channels_last)
    elif model_type == "small":
        model = torchvision.models.convnext_small(
            weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        model.to(device)
        model = model.to(memory_format=torch.channels_last)    
    if model_cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate,
            momentum=model_cfg["momentum"], weight_decay=weight_decay,
        )

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
            group["weight_decay"] = weight_decay
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