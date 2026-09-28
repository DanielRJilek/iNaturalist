import torch
from src.YOLOV1 import YOLOV1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOV1(num_classes=..., anchors=..., img_size=...).to(device)
    train_loader, val_loader = build_loaders(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    torch.save(model.state_dict(), "checkpoint.pt")

if __name__ == "__main__":
    main()