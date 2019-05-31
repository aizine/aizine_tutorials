import torch
import torch.nn as nn

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from resnet import ResNet18


class Trainer:

    def __init__(self, model, optimizer, criterion):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def epoch_train(self, train_loader):
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        epoch_loss /= len(train_loader)
        acc = 100 * correct / total
        return epoch_loss, acc

    def epoch_valid(self, valid_loader):
        self.model.eval()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        epoch_loss /= len(valid_loader)
        acc = 100 * correct / total
        return epoch_loss, acc

    @property
    def params(self):
        return self.model.state_dict()


if __name__ == '__main__':
    model = ResNet18(10)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.001,
                                momentum=0.9,
                                weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dtrain = torchvision.datasets.CIFAR10(
        './data/', train=True, transform=transform, download=True)
    dvalid = torchvision.datasets.CIFAR10(
        './data/', train=False, transform=transform, download=True)

    train_loader = DataLoader(dtrain, batch_size=config.BATCH_SIZE, shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(dvalid, batch_size=config.BATCH_SIZE)

    best_acc = -1
    for epoch in range(1, 1 + config.NUM_EPOCHS):
        train_loss, train_acc = trainer.epoch_train(train_loader)
        valid_loss, valid_acc = trainer.epoch_valid(valid_loader)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_params = trainer.params

        print(f'EPOCH: {epoch} / {num_epochs}')
        print(f'TRAIN LOSS: {train_loss:.3f}, TRAIN ACC: {train_acc:.3f}')
        print(f'VALID LOSS: {valid_loss:.3f}, VALID ACC: {valid_acc:.3f}')

    torch.save(best_params, 'weight.pth')
