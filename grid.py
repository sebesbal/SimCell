from math import sqrt
from random import randint

import PIL
import PIL.ImageDraw
import torch
from PIL.Image import Image
from matplotlib import pyplot as plt


class Node:
    def __init__(self, id):
        self.id = id
        self.edges = []
        self.influx = None
        self.fuel_cost = torch.tensor(0.0)
        self.consumed_material = torch.tensor(0.0)

    def get_edge_to(self, node):
        for e in self.edges:
            if e.dst == node:
                return e
        return None


class Edge:
    def __init__(self, source_node, dst_node, i, j):
        self.src = source_node
        self.dst = dst_node
        self.i = i
        self.j = j
        self.transported_material = torch.tensor(0.0)
        self.transport_cost = torch.tensor(sqrt(i * i + j * j))


class Grid(torch.nn.Module):
    def __init__(self, row_count, col_count, prod_count=-1):
        super().__init__()
        self.row_count = row_count
        self.col_count = col_count
        self.node_count = row_count * col_count
        self.rows = []
        self.nodes = []
        self.max_reward = 0.0
        self._init_nodes()
        if prod_count > 0:
            self._init_flux2(prod_count)
        else:
            self._init_flux()

    def _create_edge(self, a, b, i, j):
        return Edge(a, b, i, j)

    def _create_node(self, id):
        return Node(id)

    def _init_nodes(self):
        id = 0
        for row in range(self.row_count):
            row_nodes = []
            for col in range(self.col_count):
                node = self._create_node(id)
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
                            node.edges.append(self._create_edge(node, self.rows[row2][col2], i, j))

    def _init_flux(self):
        influxes = torch.rand(self.node_count)
        average = sum(influxes) / self.node_count
        consume = 0.0
        for i, node in enumerate(self.nodes):
            node.influx = influxes[i] - average
            if node.influx > 0:
                consume += node.influx.item()
        self.max_reward = consume

    def _init_flux2(self, count):
        self.max_reward = count
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

    def print_data(self):
        for row in self.rows:
            for node in row:
                print(f'( {node.id}: ' + ', '.join([f'[{e.dst.id}, {e.i}, {e.j}]' for e in node.edges]) + ')', end=' ')
            print()

    def print_stats(self):
        pass

    def draw_state(self):
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
                # if node.influx < 0:
                #     v = int(- 255 * node.consumed_material / node.influx)
                #     draw.rectangle((x, y, x + a, y + a), fill=(v, v, v))

                # v = int(- 255 * node.consumed_material / node.influx)
                # draw.rectangle((x, y, x + a, y + a), fill=(v, v, v))

                m = int(a * 0.4)
                r = (x + m, y + m, x + a - m, y + a - m)
                m2 = int(a * 0.3)
                r2 = (x + m2, y + m2, x + a - m2, y + a - m2)

                for e in node.edges:
                    weight = e.transported_material

                    if e.transport_cost > 0.0:
                        draw.line((xv, yv, xv + a/4*e.i, yv + a/4*e.j),
                                  (100, 100, int(weight * 255)), int(10 * weight))
                    else:
                        if weight > 0:
                            c = int(weight * 255 / abs(node.influx))
                            draw.rectangle(r2, fill=(c, c, c))

                if node.influx < 0:
                    draw.rectangle(r, fill=(int(- 255 * node.influx), 0, 0))
                elif node.influx > 0:
                    draw.rectangle(r, fill=(0, int(255 * node.influx), 0))

        plt.imshow(img)
        plt.show()
        plt.pause(0.1)
