from random import randint

import torch
import torch.nn.functional as F


class Node:
    def __init__(self, id, data, material):
        self.id = id
        self.edges = []  # list of tuples: (node, flux, off_i, off_j)
        self.influx = None
        self.data = data
        self.material = material


class Edge:
    def __init__(self, node, i, j, data):
        self.node = node
        self.i = i
        self.j = j
        self.data = data


class Grid(torch.nn.Module):
    def __init__(self, row_count, col_count, model, optimizer):
        super().__init__()
        self.row_count = row_count
        self.col_count = col_count
        self.node_count = row_count * col_count
        self.model_iteration_count = row_count + col_count
        self.transport_iteration_count = row_count + col_count
        self.consumed_material = torch.tensor(0.0)
        self.rows = []
        self.nodes = []
        self.model = model
        self.optimizer = optimizer
        self._init_nodes()
        self._init_flux2(2)

    def _init_nodes(self):
        id = 0
        for row in range(self.row_count):
            row_nodes = []
            for col in range(self.col_count):
                node = Node(id, torch.zeros(self.model.node_features_count), torch.tensor(0.0))
                row_nodes.append(node)
                self.nodes.append(node)
                id += 1
            self.rows.append(row_nodes)

        for row in range(self.row_count):
            for col in range(self.col_count):
                node = self.rows[row][col]
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
        consume = 0.0
        for i, node in enumerate(self.nodes):
            node.influx = influxes[i] - average
            if node.influx > 0:
                consume += node.influx.item()
        print(f"max reward: {consume * self.transport_iteration_count}")

    def _init_flux2(self, count):
        print(f"max reward: {count * self.transport_iteration_count}")
        for node in self.nodes:
            node.influx = torch.tensor(0.0)

        used = set()

        for _ in range(count):
            while True:
                row = randint(0, self.row_count - 1)
                col = randint(0, self.col_count - 1)
                node = self.rows[row][col]
                if node not in used:
                    node.influx = torch.tensor(1.0)
                    used.add(node)
                    print(f"({row}, {col}) = 1")
                    break

        for _ in range(count):
            while True:
                row = randint(0, self.row_count - 1)
                col = randint(0, self.col_count - 1)
                node = self.rows[row][col]
                if node not in used:
                    node.influx = torch.tensor(-1.0)
                    used.add(node)
                    print(f"({row}, {col}) = -1")
                    break

    def _model_iterations(self):
        for time in range(self.model_iteration_count):
            self._model_iteration()

    def _transport_iterations(self):
        for time in range(self.transport_iteration_count):
            self._produce_material()
            self._transport_iteration()
            self._consume_material()

    def _model_iteration(self):
        data_size = self.nodes[0].data.size(0)
        for a in self.nodes:
            a.weights = torch.tensor(0.0)
            a.new_data = torch.zeros(data_size)

        for a in self.nodes:
            fluxes = torch.zeros(len(a.edges))
            for i, e in enumerate(a.edges):
                b = e.node
                da, db, e.data = self.model(a, e)
                fluxes[i] = F.relu(e.data[0])
                weight_a = F.relu(da[0])
                weight_b = F.relu(db[0])
                a.new_data += da * weight_a
                b.new_data += db * weight_b
                a.weights += weight_a
                b.weights += weight_b

            fluxes = F.softmax(fluxes, dim=0)
            for i, e in enumerate(a.edges):
                e.data[0] = fluxes[i]

        for a in self.nodes:
            a.data = a.new_data / torch.clamp(a.weights, min=0.000001)

    def _transport_iteration(self):
        for a in self.nodes:
            a.new_material = torch.tensor(0.0)
        for a in self.nodes:
            for e in a.edges:
                b = e.node
                b.new_material += e.data[0] * a.material
        for i, a in enumerate(self.nodes):
            a.material = a.new_material

    def _produce_material(self):
        for a in self.nodes:
            a.material += F.relu(a.influx)

    def _consume_material(self):
        for a in self.nodes:
            consumed = torch.minimum(a.material, F.relu(-a.influx))
            a.material = a.material + consumed
            self.consumed_material = self.consumed_material + consumed

    def forward(self):
        self._model_iterations()
        self._transport_iterations()

    def run(self):
        with torch.autograd.set_detect_anomaly(True):
            while True:
                self.consumed_material = torch.tensor(0.0)
                self.optimizer.zero_grad()
                self.model.train()
                self()

                print(f"reward: {self.consumed_material.item()}")
                loss = -self.consumed_material
                loss.backward()
                self.optimizer.step()
                for a in self.nodes:
                    a.material = a.material.detach()
                    a.data = a.data.detach()
                    for e in a.edges:
                        e.data = e.data.detach()

    def print_data(self):
        for row in self.rows:
            for node in row:
                print(f'( {node.id}: ' + ', '.join([f'[{e.node.id}, {e.i}, {e.j}]' for e in node.edges]) + ')', end=' ')
            print()
