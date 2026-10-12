# eval.py
# Daniel Jilek, 2026
# Loads a trained checkpoint and scores it on the validation set. Does not train or save.

import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torchvision
from dataset import build_loaders, load_class_indices, create_mapping
from subset import prompt_order
from train import test_network

def main():
    torch.multiprocessing.set_start_method("spawn", force=True)
    batch_size_test = 64
    imageNet_mean = [0.485, 0.456, 0.406]
    imageNet_std = [0.229, 0.224, 0.225]

    while True:
        model_type = input("Choose model type [50/tiny/small]: ").strip().lower()
        if model_type in ("50", "tiny", "small"):
            break
        print("Enter 50 for ResNet50 or tiny for ConvNeXt-Tiny or small for ConvNeXt-Small.")
    order = prompt_order()

    with open("config/train.json", "r") as f:
        cfg = json.load(f)
    model_cfg = cfg["models"][model_type]
    filename = f"models/checkpoint_{model_cfg['name']}_{order}.pt"

    indices_train = load_class_indices(8, order, train=True)
    indices_valid = load_class_indices(8, order, train=False)
    map_target = create_mapping(order)
    num_classes = len(map_target.mapping_dict)
    print(f"Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        mean, std = imageNet_mean, imageNet_std
    elif model_type == "tiny":
        model = torchvision.models.convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        mean, std = imageNet_mean, imageNet_std
    elif model_type == "small":
        model = torchvision.models.convnext_small(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        mean, std = imageNet_mean, imageNet_std

    _, test_loader = build_loaders(
        mean, std, indices_train, indices_valid, map_target, batch_size_test, batch_size_test
    )

    checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    if model_type in ("tiny", "small"):
        model = model.to(memory_format=torch.channels_last)

    print(f"Loaded {filename} from epoch {int(checkpoint['epoch'])}")
    if len(checkpoint["top_1_accuracy"]) > 0:
        print(
            "Stored last val: loss {:.4f}, top-1 {:.1f}%, top-5 {:.1f}%".format(
                float(checkpoint["test_losses"][-1]),
                float(checkpoint["top_1_accuracy"][-1]),
                float(checkpoint["top_5_accuracy"][-1]),
            )
        )

    model.eval()
    test_network(test_loader, model, [], device=device)


if __name__ == "__main__":
    main()
