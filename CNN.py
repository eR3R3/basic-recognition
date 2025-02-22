import torch
import torchvision
from jinja2.compiler import F
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True, drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(320, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        return x

epoch = 100
model = CNN()
fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(15):
    model.train()
    sum = 0
    cnt = 0
    right_acc_total = 0
    for i, data in enumerate(train_loader):
        data, label = data[0], data[1]
        x = model.forward(data)
        loss = fn(x, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        sum += loss.item()
        cnt = cnt + 1
        right_num = 0
        for i in range(32):
            if x.argmax(dim=1)[i] == label[i]:
                right_num += 1
        right_acc = right_num / 32
        right_acc_total += right_acc
    print(right_acc_total / cnt)

    with torch.no_grad():
        right_num_eval = 0
        model.eval()
        cnt_eval = 0
        for data in test_loader:
            data, label = data[0], data[1]
            x = model.forward(data)
            for i in range(32):
                cnt_eval += 1
                if x.argmax(dim=1)[i] == label[i]:
                    right_num_eval += 1
        print("test accuracy", right_num_eval / cnt_eval)