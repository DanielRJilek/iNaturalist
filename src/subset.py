import torchvision
import os
import json

# This file contains code for loading class indices for a specific class from a JSON file, which can be used to subset the dataset for training and validation. The function checks if the file exists and loads the indices if it does, otherwise it returns an empty list.
def get_class_indices(class_id, root="data", version="2017"):
    # Load the full dataset temporarily (just to get categories)
    # Super-class 8 is mammals, starting around 395000-429000

    dataset = torchvision.datasets.INaturalist(
        root=root,
        version=version,
        download=False,
        transform=torchvision.transforms.ToTensor()
    )

    # 1. Load the official training metadata
    train_path = os.path.join(root, f'train_val{version}', f'train{version}.json')
    valid_path = os.path.join(root, f'train_val{version}', f'val{version}.json')

    with open(train_path, 'r') as f:
        train_metadata = json.load(f)

    with open(valid_path, 'r') as f:
        valid_metadata = json.load(f)

    # Create a set of all valid training filenames for fast lookup
    train_filenames = {os.path.basename(img['file_name']) for img in train_metadata['images']}
    valid_filenames = {os.path.basename(img['file_name']) for img in valid_metadata['images']}
    
    train_indices, valid_indices, kept_classes = [], [], []
    for idx, (cat_id, fname) in enumerate(dataset.index):
        if dataset.categories_map[cat_id]["super"] != class_id:
            continue
        if fname in train_filenames:
            train_indices.append(idx)
        elif fname in valid_filenames:
            valid_indices.append(idx)
        if cat_id not in kept_classes:
            kept_classes.append(cat_id)
    return train_indices, valid_indices, kept_classes

# Loads class indices for a specific class from a JSON file, which can be used to subset the dataset for training and validation. The function checks if the file exists and loads the indices if it does, otherwise it returns an empty list.
def load_class_indices(class_id, order):
    filename = f"data/stats/{order}_indices_train.json"
    print(f"Loading class indices for class {class_id} from {filename}...")
    if os.path.exists(filename):
        with open(filename, "r") as f:
            class_indices = json.load(f)
            return class_indices

def main():
    class_dict = {}
    class_dict["actinopterygii"] = 0
    class_dict["amphibia"] = 1
    class_dict["animalia"] = 2
    class_dict["arachnida"] = 3
    class_dict["aves"] = 4
    class_dict["chromista"] = 5
    class_dict["fungi"] = 6
    class_dict["insecta"] = 7
    class_dict["mammalia"] = 8
    class_dict["mollusca"] = 9
    class_dict["plantae"] = 10
    class_dict["protozoa"] = 11
    class_dict["reptilia"] = 12
    
    while True:
        order = input("Choose order [actinopterygii/amphibia/animalia/arachnida/aves/chromista/fungi/insecta/mammalia/mollusca/plantae/protozoa/reptilia]: ").strip().lower()
        if order in class_dict.keys():
            break
        print("Enter a valid order.")
    train_indices, valid_indices, kept_classes = get_class_indices(class_dict[order])  # Example: Get indices for class 8 (mammals)
    with open(f"data/stats/{order}_indices_train.json", "w") as f:
        json.dump(train_indices, f)
    with open(f"data/stats/{order}_indices_valid.json", "w") as f:
        json.dump(valid_indices, f)
    with open(f"data/stats/{order}_kept_classes.json", "w") as f:
        json.dump(kept_classes, f)
    return

if __name__ == "__main__":
    main()