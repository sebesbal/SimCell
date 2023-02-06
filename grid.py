import numpy as np
import torch
from numpy.random import rand


class Node:
    def __init__(self, id):
        self.id = id
        self.edges = []   # list of tuples: (node, flux, off_i, off_j)
        self.data = None


class Edge:
    def __init__(self, node, i, j, data):
        self.node = node
        self.i = i
        self.j = j
        self.data = data


class Grid:
    def __init__(self, row_count, col_count, features_count, edge_features_count):
        self.row_count = row_count
        self.col_count = col_count
        self.rows = []
        self.nodes = []
        self.features_count = features_count
        self.edge_features_count = edge_features_count

        id = 0
        for row in range(row_count):
            row_nodes = []
            for col in range(col_count):
                node = Node(id)
                node.data = torch.zeros(features_count)
                row_nodes.append(node)
                self.nodes.append(node)
                id += 1
            self.rows.append(row_nodes)

        for row in range(row_count):
            for col in range(col_count):
                node = self.rows[row][col]
                node.edges.append(Edge(node, 0, 0, torch.zeros(edge_features_count)))
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if i == 0 and j == 0:
                            continue
                        row2 = row + i
                        col2 = col + j
                        if 0 <= row2 < row_count and 0 <= col2 < col_count:
                            node.edges.append(Edge(self.rows[row2][col2], i, j, torch.zeros(edge_features_count)))

    def init_flux(self):
        n = len(self.nodes)
        influxes = rand(n)
        offset = sum(influxes) / n
        for i, node in enumerate(self.nodes):
            node.data[0] = influxes[i] - offset



