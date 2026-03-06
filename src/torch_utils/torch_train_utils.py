import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader


def pytorch_train(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: Optimizer):
    model.train()
    # todo change tuple to be variable length, for now all but last are inputs
    for batch_idx, tensors in enumerate(train_loader):
        data, target = tensors[:-1], tensors[-1]
        datums = []
        for datum in data:
            datums.append(datum.to(device))
        target = target.to(device)
        optimizer.zero_grad()
        output = model(*datums)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


def pytorch_test(model: nn.Module, device: torch.device, test_loader: DataLoader) -> float:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        n_elements = 0
        for tensors in test_loader:
            data, target = tensors[:-1], tensors[-1]
            datums = []
            for datum in data:
                datums.append(datum.to(device))
            target = target.to(device)

            output = model(*datums)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            n_elements += len(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= n_elements
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    return test_loss