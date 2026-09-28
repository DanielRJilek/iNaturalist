import torch
import torchvision
import torch.nn as nn   
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import json
from torch.utils.data import Subset
from torch.amp import autocast
from torchvision.transforms import v2


class YOLOV1(nn.Module):
    def __init__(self, num_classes, anchors, img_size, device):
        super(YOLOV1, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.img_size = img_size
        self.device = device
        
        self.num_anchors = len(anchors)
        self.num_bbox_params = 5
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.num_bbox_params = 5
        self.num_classes = num_classes

    def forward(self, x):
        pass

    def train(self, train_loader, val_loader, optimizer, criterion, device, epochs):
        self.to(device)
        train_loss = []
        val_loss = []
        best_val_loss = float('inf')
        best_model = None
        for epoch in range(epochs):
            self.train()
            train_loss = []
            for batch in train_loader:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)
    
    def test(self, test_loader, device):
        self.to(device)
        self.eval()
        test_loss = []
        for batch in test_loader:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                outputs = self(images)
                loss = criterion(outputs, targets)
                test_loss.append(loss.item())
        return test_loss

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self