import torchvision
import torch
from torch.utils.data import Subset
import torchvision.transforms.v2 as v2
import json
import os

# LabelMapper class definition, contains a mapping dictionary and a call method that takes in a label and returns the corresponding mapped value from the dictionary
class LabelMapper:
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict
        
    def __call__(self, y):
        return self.mapping_dict[y]

def as_rgb(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value, value, value]

# Loads the train and test datasets, applies the necessary transformations, takes the subset of mammal images
def load_datasets(my_dataset_mean, my_dataset_std, mammal_indices_train, mammal_indices_valid, map_target):
    # Download training data from open datasets
    training_data = torchvision.datasets.INaturalist(
        root="data",
        version="2017",
        download=True,
        transform=torchvision.transforms.Compose([
            v2.RGB(),
            torchvision.transforms.RandomResizedCrop(224), 
            torchvision.transforms.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=as_rgb(my_dataset_mean), std=as_rgb(my_dataset_std)),
            torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(1/20, 20), value=0),
        ]),
        target_transform=map_target
    )
    
    # Filter to mammal train images
    training_data_subset = Subset(training_data, mammal_indices_train)

    # Download validation data from open datasets
    test_data = torchvision.datasets.INaturalist(
        root="data",
        version="2017",
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            v2.RGB(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=as_rgb(my_dataset_mean), std=as_rgb(my_dataset_std))
        ]),
        target_transform=map_target
    )
    
    # Filter to mammal valid images
    test_data_subset = Subset(test_data, mammal_indices_valid)

    print(f"Number of training samples: {len(training_data_subset)}")
    print(f"Number of validation samples: {len(test_data_subset)}")
    return training_data_subset, test_data_subset

# Loads the mean and standard deviation of the dataset from a file for normalization
def load_dataset_stats(filename="data/stats/dataset_stats.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()
        my_dataset_mean = float(lines[0].split(":")[1].strip())
        my_dataset_std = float(lines[1].split(":")[1].strip())
    print(f"Using dataset mean: {my_dataset_mean} and dataset standard deviation: {my_dataset_std} for normalization.")
    return my_dataset_mean, my_dataset_std

# Creates a mapping from the original class indices to new class indices based on the kept classes and returns a LabelMapper object that can be used as a target_transform in the dataset to remap the class labels to a contiguous range for training the model on a subset of classes.
def create_mapping():
    with open("data/stats/kept_classes.json", "r") as f:
        kept_classes = json.load(f)
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_classes)}
    map_target = LabelMapper(mapping)
    return map_target

def build_loaders(my_dataset_mean, my_dataset_std, mammal_indices_train, mammal_indices_valid, map_target, batch_size_train, batch_size_test):
    train_data, test_data = load_datasets(my_dataset_mean, my_dataset_std, mammal_indices_train, mammal_indices_valid, map_target)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, test_loader

# Loads class indices for a specific class from a JSON file, which can be used to subset the dataset for training and validation. The function checks if the file exists and loads the indices if it does, otherwise it returns an empty list.
def load_class_indices(class_id, filename="data/stats/mammal_indices_train.json"):
    print(f"Loading class indices for class {class_id} from {filename}...")
    if os.path.exists(filename):
        with open(filename, "r") as f:
            class_indices = json.load(f)
            return class_indices