import torch
import torch.nn.functional as F
from torch.amp import autocast

# trains the model on the training data and updates the training losses and counters
def train_network( dataloader, model, optimizer, train_losses, train_counter, epoch, log_interval = 10, device="cpu", scaler=None ):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device, non_blocking=True, memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            output = model(data)
            loss = F.cross_entropy(output, target, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*len(data)) + ((epoch-1)*len(dataloader.dataset)))
    del data, target, output, loss
    return

# tests the model on the test data and prints the average loss and accuracy
def test_network( dataloader, model, test_losses, device="cpu" ):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device, non_blocking=True, memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            with autocast(device_type=device.type):
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()

                pred = output.argmax(dim=1, keepdim=True)
                correct_top1 += pred.eq(target.view_as(pred)).sum().item()              
               
                # Get indices of the 5 largest values
                _, top5_preds = output.topk(5, dim=1, largest=True, sorted=True)
                # Expand target to match the shape of top5_preds [batch_size, 5]
                target_expanded = target.view(-1, 1).expand_as(top5_preds)
                # Check if target is in any of the 5 positions
                correct_top5 += top5_preds.eq(target_expanded).sum().item()
            del data, target, output
        test_loss /= len(dataloader.dataset)
        test_losses.append(test_loss)
    total = len(dataloader.dataset)
    top_1_accuracy = 100. * correct_top1 / total
    top_5_accuracy = 100. * correct_top5 / total
    print('\nTest set: Avg. loss: {:.4f}'.format(test_loss))
    print('Top-1 Accuracy: {}/{} ({:.1f}%)'.format(correct_top1, total, top_1_accuracy))
    print('Top-5 Accuracy: {}/{} ({:.1f}%)\n'.format(correct_top5, total, top_5_accuracy))
    return top_1_accuracy, top_5_accuracy

def run_training(model, train_loader, test_loader, optimizer, epochs, device, test_interval=1, scheduler=None, start_epoch=0):
    train_losses, train_counter, test_losses = [], [], []
    top_1_accuracy, top_5_accuracy = [], []
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        train_network(
            train_loader, model, optimizer,
            train_losses, train_counter, epoch, device=device,
            scaler=scaler,
        )
        if epoch % test_interval == 0:
            top_1, top_5 = test_network(test_loader, model, test_losses, device=device)
            top_1_accuracy.append(top_1)
            top_5_accuracy.append(top_5)
            if scheduler is not None:
                scheduler.step(test_losses[-1])
    return train_losses, test_losses, top_1_accuracy, top_5_accuracy