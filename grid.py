import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import rand


class Node:
    def __init__(self, id, data):
        self.id = id
        self.edges = []   # list of tuples: (node, flux, off_i, off_j)
        self.influx = None
        self.data = data
        self.material = torch.tensor(0.0, requires_grad=True)


class Edge:
    def __init__(self, node, i, j, data):
        self.node = node
        self.i = i
        self.j = j
        self.data = data


class Grid:
    def __init__(self, row_count, col_count, model, optimizer):
        self.row_count = row_count
        self.col_count = col_count
        self.node_count = row_count * col_count
        self.iterations = row_count + col_count
        self.consumed_material = torch.tensor(0.0, requires_grad=True)
        self.rows = []
        self.nodes = []
        self.model = model
        self.optimizer = optimizer
        self._init_nodes()
        self._init_flux()

    def _init_nodes(self):
        id = 0
        for row in range(self.row_count):
            row_nodes = []
            for col in range(self.col_count):
                node = Node(id, torch.zeros(self.model.node_features_count))
                row_nodes.append(node)
                self.nodes.append(node)
                id += 1
            self.rows.append(row_nodes)

        for row in range(self.row_count):
            for col in range(self.col_count):
                node = self.rows[row][col]
                node.edges.append(Edge(node, 0, 0, torch.zeros(self.model.edge_features_count)))
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        row2 = row + i
                        col2 = col + j
                        if 0 <= row2 < self.row_count and 0 <= col2 < self.col_count:
                            node.edges.append(Edge(self.rows[row2][col2], i, j,
                                                   torch.zeros(self.model.edge_features_count)))

    def _init_flux(self):
        influxes = torch.rand(self.node_count)
        average = sum(influxes) / self.node_count
        for i, node in enumerate(self.nodes):
            node.influx = influxes[i] - average

    def _apply_model(self):
        data_size = self.nodes[0].data.size(0)
        # weights = torch.zeros(self.node_count, requires_grad=True)
        weights = []
        new_node_data = []
        for _ in range(self.node_count):
            weights.append(torch.tensor(0.0, requires_grad=True))
            new_node_data.append(torch.tensor(0.0, requires_grad=True))
        # weights = [None] * self.node_count
        # new_node_data = torch.zeros((self.node_count, data_size), requires_grad=True)

        # new_node_data = []
        # for i in range(self.node_count):
        #    new_node_data.append(torch.zeros(data_size))

        for a in self.nodes:
            # fluxes = torch.zeros(len(a.edges), requires_grad=True)
            fluxes = []
            for i, e in enumerate(a.edges):
                b = e.node
                data_a, data_b, e.data = self.model(a, e)
                fluxes.append(F.relu(e.data[0]))
                weight_a = F.relu(data_a[0])
                weight_b = F.relu(data_b[0])
                weights[a.id].add_(weight_a)
                weights[b.id].add_(weight_b)
                new_node_data[a.id].add_(data_a * weight_a)
                new_node_data[b.id].add_(data_b * weight_b)
            for i, e in enumerate(a.edges):
                e.data[0] = fluxes[i]

        for i, a in enumerate(self.nodes):
            a.data = new_node_data[i] / torch.clamp(weights[i], min=0.000001)

    def _move_material(self):
        new_material = torch.zeros(self.node_count)
        for a in self.nodes:
            material = a.material
            for e in a.edges:
                b = e.node
                new_material[b.id] += e.data[0] * material
        for i, a in enumerate(self.nodes):
            a.material = new_material[i]

    def _produce_material(self):
        for a in self.nodes:
            if a.influx > 0:
                a.material.add_(a.influx)

    def _consume_material(self):
        for a in self.nodes:
            if a.influx < 0:
                consumed = min(a.material, -a.influx)
                a.material -= consumed
                self.consumed_material.add_(consumed)

    def run(self):
        with torch.autograd.set_detect_anomaly(True):
            while True:
                self.optimizer.zero_grad()
                self.model.train()
                self.consumed_material = torch.tensor(0.0, requires_grad=True)
                for _ in range(self.iterations):
                    self._apply_model()

                for _ in range(self.iterations):
                    self._produce_material()
                    self._move_material()
                    self._consume_material()

                print(f"consumed_material: {self.consumed_material.item()}")
                loss = -self.consumed_material
                loss.backward()
                # F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
                self.optimizer.step()

    def print_data(self):
        for row in self.rows:
            for node in row:
                print(f'( {node.id}: ' + ', '.join([f'[{e.node.id}, {e.i}, {e.j}]' for e in node.edges]) + ')', end=' ')
            print()
