import os.path
import random

import torch
from torch import optim
from tqdm import tqdm

from grid import Grid
from model import Model


def test_grid():
    model = Model(20, 20)
    if os.path.exists('SimCell.p'):
        model.load_state_dict(torch.load('SimCell.p'))

    optimizer = optim.Adam(model.parameters())  # , lr=0.01)  #, betas=(0.999, 0.999))
    for i in range(30):
        size = 2 + i // 10
        prod = size - 1
        grid = Grid(size, size, prod, model, optimizer)
        grid.draw_state()
        grid.run()


def test_grid2():
    model = Model(10, 10)
    if os.path.exists('SimCell.p'):
        model.load_state_dict(torch.load('SimCell.p'))

    optimizer = optim.Adam(model.parameters())  # , lr=0.01)  #, betas=(0.999, 0.999))
    optimizer.loss = None
    min_size = 2
    max_size = 4
    grids = []
    for size in range(min_size, max_size + 1):
        for i in range(50):
            prod = size - 1
            grid = Grid(size, size, prod, model, optimizer)
            grids.append(grid)

    random.shuffle(grids)

    for epoch in range(100):
        print(epoch)
        # for i, grid in enumerate(tqdm(grids)):
        for i, grid in enumerate(grids):
            # grid.run(print_stats=(i == len(grids) - 1))
            grid.run(backprop=True, print_stats=(i % 10 == 0))
        torch.save(model.state_dict(), 'SimCell.p')


if __name__ == '__main__':
    test_grid2()
