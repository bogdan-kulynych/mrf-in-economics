import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpMaximize, LpVariable, value)


def find_optimal_strategy(states, controls, costs, kernels, solver=None):
    """
    :param states: Number of system states (X)
    :param controls: Number of system controls (U)
    :param kernels: Transition kernels. Dimensionality X x X x U
    """
    tolerance = 10e-15

    X = range(states)
    U = range(controls)
    R = costs
    Q = kernels

    # Check costs
    # Check num of rows
    assert(len(R) == states)
    for row in R:
        # Check num of cols
        assert(len(row) == controls)

    # Check kernels
    # Check num of rows
    assert(len(Q) == states)
    for row in Q:
        # Check num of cols
        assert(len(row) == states)
        for items in row:
            # Check num of items
            assert(len(items) == controls)
        # Check if distribution is normed
        for dist in zip(*row):
            assert(sum(dist)-1 < tolerance)

    # LP object
    optm = LpProblem("Optimal strategy", sense=LpMinimize)

    # Variables (grouped by x)
    vars = {x: [LpVariable("({},{})".format(x, u), 0, 1) \
                for u in U] for x in X}

    # Variables (flattened)
    Z = []
    for x in X:
        Z.extend(vars[x])

    # Objective
    optm.objective = sum(np.dot(vars[x], R[x]) for x in X)

    # Constraints
    for x in X:
        cn = (sum(vars[x]) == sum(Q[y][x][u]*vars[y][u] for u in U for y in X))
        optm.add(cn)
    cn = sum(vars[x][u] for u in U for x in X) == 1
    optm.add(cn)

    optm.solve(solver)

    return [str(z) for z in Z if value(z) != 0]


if __name__ == '__main__':
    states = 4
    controls = 4
    kernels = np.array([
        [[0.1, 0.2, 0.7, 0.7], [0.1, 0.1, 0.1, 0.1], [0.4, 0.3, 0.1, 0.2], [0.4, 0.4, 0.1, 0  ]],
        [[0.7, 0.1, 0.2, 0.5], [0.1, 0  , 0.2, 0.2], [0.1, 0  , 0.5, 0.1], [0.1, 0.9, 0.1, 0.2]],
        [[0.3, 0.3, 0.9, 0.1], [0.1, 0.5, 0  , 0.7], [0.2, 0  , 0.1, 0.1], [0.4, 0.2, 0  , 0.1]],
        [[0.1, 0.1, 0  , 0.8], [0.5, 0.1, 0.6, 0.1], [0.2, 0.7, 0.2, 0  ], [0.2, 0.1, 0.2, 0.1]]
    ])
    costs = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ])

    strategy = find_optimal_strategy(states, controls, costs, kernels)
    for control in sorted(strategy):
        print(control)

