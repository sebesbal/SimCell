import os.path

import torch
from torch import optim

from grid import Grid
from model import Model


def test_grid():
    model = Model(10, 10)
    if os.path.exists('SimCell.p'):
        model.load_state_dict(torch.load('SimCell.p'))

    optimizer = optim.Adam(model.parameters())  #, lr=0.01)  #, betas=(0.999, 0.999))
    for i in range(30):
        size = 3 + i // 10
        prod = size - 1
        grid = Grid(size, size, prod, model, optimizer)
        grid.draw_state()
        grid.run()


if __name__ == '__main__':
    test_grid()
