from torch import optim

from grid import Grid
from model import Model


def test_grid():
    model = Model(5, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    grid = Grid(3, 3, model, optimizer)
    grid.run()


if __name__ == '__main__':
    test_grid()
