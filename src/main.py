import torch
from src.YOLOV1 import YOLOV1
from src.dataset import build_loaders, load_dataset_stats, load_class_indices, create_mapping
from src.train import run_training

def main():
    batch_size_train = 64
    batch_size_test = 500
    epochs = 20
    random_seed = 1
    learning_rate = 1e-4    
    momentum = 0.9  
    weight_decay=5e-4
    test_interval = 1
    patience = 2
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True

    my_dataset_mean, my_dataset_std = load_dataset_stats()
    # Get mammal indices for both train and valid sets
    mammal_indices_train = load_class_indices(8, "mammal_indices_train.json")
    mammal_indices_valid = load_class_indices(8, "mammal_indices_test.json")
    map_target = create_mapping()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOV1(num_classes=186, initial_kernel_size=7).to(device)
    train_loader, test_loader = build_loaders(my_dataset_mean, my_dataset_std, mammal_indices_train, mammal_indices_valid, map_target, batch_size_train, batch_size_test)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)
    run_training(model, train_loader, test_loader, optimizer, scheduler, epochs, device, test_interval)
    torch.save(model.state_dict(), "checkpoint.pt")

if __name__ == "__main__":
    main()