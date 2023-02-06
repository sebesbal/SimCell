import torch


class Model(torch.nn.module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.linear_1 = torch.nn.Linear(num_in, num_out)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()

    def forward(self, a, b):
        x = torch.cat((a, b), 0)
        x = self.linear_1(x)
        x = torch.sigmoid(x)
        return x

