from grid import Grid


def test_grid():
    grid = Grid(3, 3, 5, 5)
    for row in grid.rows:
        for node in row:
            print(f'( {node.id}: ' + ', '.join([f'[{e.node.id}, {e.i}, {e.j}]' for e in node.edges]) + ')', end=' ')
        print()


if __name__ == '__main__':
    test_grid()

