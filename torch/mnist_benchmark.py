import time
import argparse

import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms


class MLPNet(nn.Module):
    def __init__(self, hidden_size: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.net(x), dim = 1)


def train_one_epoch(model, device, dataloader, optimizer, epoch, log_interval: int = 100):
    model.train()
    total_loss = 0.0
    start_time = time.perf_counter()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    duration = time.perf_counter() - start_time
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} completed in {duration:.2f}s | Avg Loss: {avg_loss:.4f}")
    return duration, avg_loss


def evaluate(model, device, dataloader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    print(f"Test set: Avg loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return test_loss, accuracy

def get_data_loaders(batch_size:int  = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train = True, download = True, transform = transform),
        batch_size = batch_size,
        shuffle = True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train = False, transform = transform),
        batch_size = batch_size,
        shuffle = False
    )

    return train_loader, test_loader


def select_device(preferred = None):
    if preferred:
        preferred = preferred.lower()
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred == "cpu":
            return torch.device("cpu")

    # fallback
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_optimizer(optimizer_name, model, lr, momentum = 0.5):
    name = optimizer_name.lower()
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def main(args):
    device = select_device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders(args.batch_size)
    model = MLPNet(hidden_size=args.hidden_size).to(device)
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.momentum)

    total_time = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_time, _ = train_one_epoch(model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)
        total_time += epoch_time

    print(f"\nTraining completed in {total_time:.2f}s total ({total_time/args.epochs:.2f}s/epoch)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train MNIST (MLP) with PyTorch")
    parser.add_argument("--epochs", type = int, default = 5, help = "Number of training epochs")
    parser.add_argument("--batch-size", type = int, default = 128, help = "Batch size")
    parser.add_argument("--hidden-size", type = int, default = 128, help = "Hidden layer size")
    parser.add_argument("--lr", type = float, default = 0.01, help = "Learning rate")
    parser.add_argument("--momentum", type = float, default = 0.5, help = "Momentum for SGD")
    parser.add_argument("--optimizer", type = str, default = "sgd", help = "Optimizer: sgd | adam | rmsprop")
    parser.add_argument("--device", type = str, default = None, help = "Device: mps | cuda | cpu")

    args = parser.parse_args()
    main(args)
