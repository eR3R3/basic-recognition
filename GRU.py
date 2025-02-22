import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True, drop_last=True)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.seq_length = seq_length
        self.lstm = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size*seq_length, out_features=output_size)

    def forward(self, x):
        x = x.reshape(-1, 28, self.input_size)
        x, _ = self.lstm(x)
        x = x.reshape(-1, self.hidden_size*self.seq_length)
        x = self.linear(x)
        return x


model = GRU(input_size=28, hidden_size=32, output_size=10, seq_length=28)
fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(10):
    model.train()
    sum = 0
    cnt = 0
    right_acc_total = 0
    for i, data in enumerate(train_loader):
        data, label = data[0], data[1]
        data = data.reshape(-1, 28, 28)
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








