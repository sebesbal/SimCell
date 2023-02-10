import msvcrt
from math import sqrt
from random import randint
from time import sleep

import PIL
import numpy
import torch
import torch.nn.functional as F
import PIL.ImageDraw
from PIL.Image import Image
from matplotlib import pyplot as plt


class Node:
    def __init__(self, id, data, material):
        self.id = id
        self.edges = []  # list of tuples: (node, flux, off_i, off_j)
        self.influx = None
        self.data = data
        self.material = material
        self.fuel_cost = torch.tensor(0.0)
        self.consumed_material = torch.tensor(0.0)


class Edge:
    def __init__(self, node, i, j, data):
        self.node = node
        self.i = i
        self.j = j
        self.data = data
        self.transported_material = torch.tensor(0.0)
        self.length = torch.tensor(sqrt(i * i + j * j))


class Grid(torch.nn.Module):
    def __init__(self, row_count, col_count, prod_count, model, optimizer):
        super().__init__()
        self.row_count = row_count
        self.col_count = col_count
        self.node_count = row_count * col_count
        self.model_iteration_count = 3 * (row_count + col_count - 2)
        self.transport_iteration_count = row_count + col_count - 2
        self.consumed_material = torch.tensor(0.0)
        self.fuel_cost = torch.tensor(0.0)
        self.rows = []
        self.nodes = []
        self.model = model
        self.optimizer = optimizer
        self.max_reward = 0.0
        self._init_nodes()
        self._init_flux()
        # self._init_flux2(prod_count)
        print(f"max reward: {self.max_reward}")

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
        self.max_reward = consume * self.transport_iteration_count

    def _init_flux2(self, count):
        self.max_reward = count * self.transport_iteration_count
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
            a.new_fuel_cost = torch.tensor(0.0)
        for a in self.nodes:
            for e in a.edges:
                b = e.node
                transported_material = e.data[0] * a.material
                e.transported_material += transported_material
                b.new_material += transported_material
                fuel_cost = transported_material * (e.length + a.fuel_cost)
                b.new_fuel_cost += fuel_cost
                self.fuel_cost += fuel_cost
        for i, a in enumerate(self.nodes):
            a.material = a.new_material
            a.fuel_cost = a.new_fuel_cost

    def _produce_material(self):
        for a in self.nodes:
            a.material += F.relu(a.influx)

    def _consume_material(self):
        for a in self.nodes:
            consumed = torch.minimum(a.material, F.relu(-a.influx))
            # self.fuel_cost = self.fuel_cost + a.fuel_cost * consumed / a.material
            a.material = a.material - consumed
            a.consumed_material += consumed
            self.consumed_material = self.consumed_material + consumed

    def forward(self):
        self._model_iterations()
        self._transport_iterations()

    def run(self):
        with torch.autograd.set_detect_anomaly(False):
            for i in range(20):
                self.consumed_material = torch.tensor(0.0)
                self.fuel_cost = torch.tensor(0.0)
                self.optimizer.zero_grad()
                self.model.train()
                self()

                # print(f"reward: {100 * self.consumed_material.item() / self.max_reward}%")

                loss = - (5 * self.consumed_material - self.fuel_cost)

                print(f"consumed: {100 * self.consumed_material.item() / self.max_reward:.2f}%"
                      f",  fuel_cost: {self.fuel_cost:.2f},  reward: {-loss:.2f}")

                loss.backward()
                self.optimizer.step()

                full_material = 0.0
                self.draw_state()
                for a in self.nodes:
                    full_material += a.material
                    a.material = a.material.detach()
                    a.data = a.data.detach()

                    #if a.influx < 0:
                    #    print(f'consumed material: {a.consumed_material.item()}')
                    a.consumed_material = torch.tensor(0.0)
                    a.material = torch.tensor(0.0)
                    a.fuel_cost = torch.tensor(0.0)
                    a.data = torch.zeros(self.model.node_features_count)
                    for e in a.edges:
                        e.data = e.data.detach()
                        e.data = torch.zeros(self.model.edge_features_count)
                        e.transported_material = torch.tensor(0.0)

                # print(f'full material: {full_material}')

                if msvcrt.kbhit():
                    print("you pressed", msvcrt.getch(), "so now i will quit")
        torch.save(self.model.state_dict(), 'SimCell.p')

    def print_data(self):
        for row in self.rows:
            for node in row:
                print(f'( {node.id}: ' + ', '.join([f'[{e.node.id}, {e.i}, {e.j}]' for e in node.edges]) + ')', end=' ')
            print()

    def draw_state(self):
        # for i in range(10):
        #     img = PIL.Image.new(mode="RGB", size=(200, 200))
        #     draw = PIL.ImageDraw.Draw(img)
        #     draw.line((10*i, 0) + img.size, fill=128)
        #     draw.line((0, img.size[1], img.size[0], 0), fill=128)
        #     # img.show('fasza')
        #     plt.ion()
        #     plt.imshow(img)
        #     plt.show()
        #     # sleep(0.2)
        #     plt.pause(0.01)
        #     plt.clf()  # will make the plot window empty
        #     # plt.close()
        #     img.close()

        plt.ion()
        plt.clf()
        a = 100  # cell size
        img = PIL.Image.new(mode="RGB", size=(self.col_count * a, self.row_count * a))
        draw = PIL.ImageDraw.Draw(img)
        for i, row in enumerate(self.rows):
            for j, node in enumerate(row):
                x = i * a
                y = j * a
                xv = x + a/2
                yv = y + a/2
                if node.influx < 0:
                    v = int(- 255 * node.consumed_material / (self.transport_iteration_count * node.influx))
                    draw.rectangle((x, y, x + a, y + a), fill=(v, v, v))

                for e in node.edges:
                    weight = e.transported_material
                    draw.line((xv, yv, xv + a/4*e.i, yv + a/4*e.j),
                              (100, 100, int(weight * 255)), int(10 * weight))
                    # if i == 1 and j == 1:
                    #    print(f'weight: {weight}')

                m = int(a * 0.4)
                r = (x + m, y + m, x + a - m, y + a - m)
                if node.influx < 0:
                    draw.rectangle(r, fill=(int(- 255 * node.influx), 0, 0))
                elif node.influx > 0:
                    draw.rectangle(r, fill=(0, int(255 * node.influx), 0))

                    # e.data[0]

        plt.imshow(img)
        plt.show()
        plt.pause(0.1)