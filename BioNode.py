import random

import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.optim.lr_scheduler import ExponentialLR


class BioNode(torch.nn.Module):
    def __init__(self, input_count, hidden_count, output_count, depth):
        super().__init__()
        self.depth = depth
        self.layers = torch.nn.ModuleList()
        n = 1
        last_layer = n * input_count
        for i in range(depth):
            next_layer = output_count if i == depth - 1 else hidden_count
            layer = torch.nn.Linear(last_layer, next_layer)
            layer.weight.detach().normal_(0.0, 0.1)
            # layer.weight.detach().zero_()
            layer.bias.detach().normal_(0.0, 0.1)
            self.layers.append(layer)
            last_layer = n * next_layer
        # self.layers[0].weight.data[0][1] = 2.0
        # self.layers[1].weight.data[0][2] = 1.0

    def forward(self, x):
        for layer in self.layers:
            # x_ln = x*x  # torch.log(x)
            # x_ln = torch.log(torch.relu(x).clamp(min=0.0001))
            # x_exp = torch.exp(x)
            # x = torch.cat([x, x_ln, x_exp], dim=1)
            # x = torch.stack([x, x_ln], dim=1)
            x = layer(x)
            # x = torch.relu(x)
            x = torch.sigmoid(x)
        return x


def bio1(f):
    model = BioNode(1, 3, 1, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    train_losses = []
    loss_fn = nn.MSELoss(reduction='mean')

    plt.figure(1, figsize=(10, 10))
    plt.subplot(211)
    plt.title("Training and Validation Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    # plt.figure(2, figsize=(10, 5))
    plt.subplot(212)
    plt.title("Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    for epoch in range(1, 1000):
        model.train()
        optimizer.zero_grad()
        sum = 0.0

        batches = 1000
        for i in range(batches):
            x = 10 * torch.rand(10).view(-1, 1)
            target = f(x)
            y = model(x)
            # loss = torch.pow(y - target, 2)
            # loss = loss.sum() / 10.0
            loss = loss_fn(y, target)
            loss.backward()
            print(loss.item())
            sum += loss.item()

        optimizer.step()
        scheduler.step()

        train_losses.append(sum / batches)
        print(sum / batches)

        plt.ion()
        plt.clf()

        plt.subplot(211)
        plt.plot(train_losses, label="train")

        model.eval()
        ys = []
        for i in range(100):
            x = i / 10.0
            ys.append(model(torch.tensor([[x]])).item())
        plt.subplot(212)
        plt.plot(ys, label="Function")

        plt.show()
        plt.pause(0.01)


if __name__ == "__main__":
    bio1(lambda x: x*x)
