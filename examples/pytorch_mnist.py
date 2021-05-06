"""
Adapted from the following PyTorch example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import oh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


@oh.register("MNISTNet")  # name optional
class Net(nn.Module):
    def __init__(self, chans1, chans2, chans3, dropout):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, chans1, 3, 1)
        self.conv2 = nn.Conv2d(chans1, chans2, 3, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(144 * chans2, chans3)
        self.fc2 = nn.Linear(chans3, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % oh.config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



oh.config.load_str(
"""
[training]
batch_size = 64
epochs = 10
lr = 1.0
gamma = 0.7
seed = 1
log_interval = 10

[training.net]
@call = MNISTNet
chans1 = 32
chans2 = 64
chans3 = 128
dropout = 0.5

[training.optimizer]
@call = torch.optim:Adadelta
lr = ${training.lr}

[training.scheduler]
@call = torch.optim.lr_scheduler:StepLR
step_size = 1
gamma = ${training.gamma}

[testing]
batch_size = 1024

[system]
use_cuda = true
""")

use_cuda = oh.config.system.use_cuda and torch.cuda.is_available()

torch.manual_seed(oh.config.training.seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.MNIST('data', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('data', train=False,
                    transform=transform)

if use_cuda:
    device = torch.device("cuda")
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
else:
    device = torch.device("cpu")
    cuda_kwargs = {}

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=oh.config.training.batch_size, **cuda_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=oh.config.testing.batch_size, **cuda_kwargs)

with oh.config.enter("training"):
    model = oh.config.net().to(device)
    optimizer = oh.config.optimizer(model.parameters())
    scheduler = oh.config.scheduler(optimizer)

    for epoch in range(1, oh.config.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
