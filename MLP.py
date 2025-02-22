import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True, drop_last=True)

class MLP(nn.Module):
    def __init__(self, in_features, intermediate_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, intermediate_features)
        self.fc2 = nn.Linear(intermediate_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

model = MLP(in_features=28*28, intermediate_features=256, out_features=10)
fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):
    model.train()
    sum = 0
    cnt = 0
    right_acc_total = 0
    for i, data in enumerate(train_loader):
        data, label = data[0], data[1]
        data = data.reshape(-1, 28*28)
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
    print(right_acc_total/cnt)

    with torch.no_grad():
        right_num_eval = 0
        model.eval()
        cnt_eval = 0
        for data in test_loader:
            data, label = data[0], data[1]
            data = data.reshape(-1, 28*28)
            x = model.forward(data)
            for i in range(32):
                cnt_eval += 1
                if x.argmax(dim=1)[i] == label[i]:
                    right_num_eval += 1
        print("test accuracy", right_num_eval / cnt_eval)








