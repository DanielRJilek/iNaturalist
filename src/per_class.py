import json
import torch
import torchvision
from dataset import build_loaders, load_class_indices, create_mapping

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while True:
        order = input("Choose order [actinopterygii/amphibia/animalia/arachnida/aves/chromista/fungi/insecta/mammalia/mollusca/plantae/protozoa/reptilia]: ").strip().lower()
        if order in ("actinopterygii", "amphibia", "animalia", "arachnida", "aves", "chromista", "fungi", "insecta", "mammalia", "mollusca", "plantae", "protozoa", "reptilia"):
            break
        print("Enter a valid order.")
    train_idx = load_class_indices(8, f"data/stats/{order}_indices_train.json")
    valid_idx = load_class_indices(8, f"data/stats/{order}_indices_valid.json")
    _, test_loader = build_loaders(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        train_idx, valid_idx, create_mapping(order), 64, 500,
    )
    num_classes = len(create_mapping(order).mapping_dict)
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(f"models/checkpoint_resnet50_{order}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            total += torch.bincount(target, minlength=num_classes).cpu()
            hit = target[pred == target]
            correct += torch.bincount(hit, minlength=num_classes).cpu()

    with open(f"data/stats/{order}_kept_classes.json") as f:
        kept = json.load(f)
    categories = test_loader.dataset.dataset.all_categories

    rows = []
    for c in range(num_classes):
        acc = 100.0 * correct[c] / total[c]
        name = str(categories[kept[c]]).replace("\\", "/").split("/")[-1]
        rows.append((acc, int(correct[c]), int(total[c]), name))

    for acc, k, n, name in sorted(rows):
        print(f"{acc:6.1f}%  {k:4d}/{n:<4d}  {name}")

if __name__ == "__main__":
    main()