# plot.py
# Daniel Jilek, 2026
# This file contains code for plotting the training and validation loss and accuracy for a specific model and order.

import torch
import matplotlib.pyplot as plt
import numpy as np
from subset import prompt_order

# Plots the training and validation loss and accuracy for a specific model and order.
def main():
    while True:
        model_type = input("Choose model type [y/18/50/tiny/small]: ").strip().lower()
        if model_type in ("y", "18", "50", "tiny", "small"):
            break
        print("Enter y for YOLOV1 or 18 for ResNet18 or 50 for ResNet50 or tiny for TinyNet or small for SmallNet.")

    order = prompt_order()

    if model_type == "y":
        model_name = "YOLOV1"
        filename = f"models/checkpoint_yolo_{order}.pt"
    elif model_type == "18":
        model_name = "ResNet18"
        filename = f"models/checkpoint_resnet18_{order}.pt"
    elif model_type == "50":
        model_name = "ResNet50"
        filename = f"models/checkpoint_resnet50_{order}.pt"
    elif model_type == "tiny":
        model_name = "ConvNeXt-Tiny"
        filename = f"models/checkpoint_convnext_tiny_{order}.pt"
    elif model_type == "small":
        model_name = "ConvNeXt-Small"
        filename = f"models/checkpoint_convnext_small_{order}.pt"

    checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
    test_loss = checkpoint["test_losses"].tolist()
    top1 = checkpoint["top_1_accuracy"].tolist()
    top5 = checkpoint["top_5_accuracy"].tolist()
    train_loss = checkpoint["train_losses"].tolist()

    epochs = range(1, len(test_loss) + 1)
    points_per_epoch = len(train_loss) / len(test_loss)
    train_epochs = [(i + 1) / points_per_epoch for i in range(len(train_loss))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(train_epochs, train_loss, label="train", alpha=0.7)
    axes[0].plot(epochs, test_loss, marker="o", label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs, top1, marker="o", label="top-1")
    axes[1].plot(epochs, top5, marker="o", label="top-5")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[0].grid(axis="y")
    axes[1].grid(axis="y")
    axes[1].legend()
    last = len(test_loss)
    ticks = range(0, last + 6, 5)
    for ax in axes:
        ax.set_xlim(0, last)
        ax.set_xticks(ticks)
    model_name = model_name.capitalize()
    order = order.capitalize()
    fig.suptitle(f"{model_name} on {order}")
    fig.tight_layout()
    fig.savefig(f"results/{model_name}_{order}.png")
    plt.show()

if __name__ == "__main__":
    main()