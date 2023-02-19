from math import sqrt

import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, node_features_count, edge_features_count):
        super().__init__()
        self.node_features_count = node_features_count
        self.edge_features_count = edge_features_count
        num_in = 2 * (node_features_count + 2) + edge_features_count + 1
        # num_out = node_features_count + edge_features_count
        num_out = 2 * node_features_count + edge_features_count

        self.linear_1 = torch.nn.Linear(num_in, num_in)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        self.linear_2 = torch.nn.Linear(num_in, num_out)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()

        self.linear_2_1 = torch.nn.Linear(num_in, num_in)
        self.linear_2_1.weight.detach().normal_(0.0, 0.1)
        self.linear_2_1.bias.detach().zero_()
        self.linear_2_2 = torch.nn.Linear(num_in, num_out)
        self.linear_2_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2_2.bias.detach().zero_()

    def forward(self, a, e):
        b = e.node
        # r = torch.tensor(float(e.i * e.i + e.j * e.j))
        r = e.length

        x = torch.cat((r.view(1), a.data, a.material.view(1), a.influx.view(1),
                       b.data, b.material.view(1), b.influx.view(1), e.data), 0)

        if a == e.node:
            x = self.linear_1(x)
            x = torch.relu(x)
            x = self.linear_2(x)
            x = torch.relu(x)
        else:
            x = self.linear_2_1(x)
            x = torch.relu(x)
            x = self.linear_2_2(x)
            x = torch.relu(x)

        n = self.node_features_count

        if False:
            b_data = x[0:n].clone()
            e_data = x[n:].clone()
            # e_data = x[2*n:].clone()
            return b_data, e_data
        else:
            b_data = x[0:n].clone()
            a_data = x[n:2*n].clone()
            e_data = x[2*n:].clone()
            return a_data, b_data, e_data


