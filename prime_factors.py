from enum import Enum
import numpy as np


class PrimeFactorTree:
    def __init__(self, value: int, left, right):
        self.value = value
        self.left = left
        self.right = right
        self.is_even = self.value % 2 == 0

    def __str__(self):
        if self.left is None and self.right is None:
            return str(self.value)
        else:
            return "(%d = %s * %s)" % (self.value, self.left, self.right)


class PrimeFactorTreeStates(Enum):
    VALUE = 0
    IS_EVEN = 1
    LEFT_OPT = 2
    RIGHT_OPT = 3


def get_trees(n: int) -> [PrimeFactorTree]:
    return [__get_prime_factor_tree(x) for x in range(2, n + 2)]


def get_numpy_arrays(n: int) -> [np.array]:
    return [__tree_array_to_numpy_array(__tree_to_tree_array(tree)) for tree in get_trees(n)]


def __tree_to_tree_array(tree: PrimeFactorTree) -> []:
    tree_array = []

    tree_array.append([PrimeFactorTreeStates.VALUE, tree.value])
    tree_array.append([PrimeFactorTreeStates.IS_EVEN, tree.is_even])

    tree_array.append([PrimeFactorTreeStates.LEFT_OPT, tree.left is not None])
    if tree.left is not None:
        tree_array.extend(__tree_to_tree_array(tree.left))

    tree_array.append([PrimeFactorTreeStates.RIGHT_OPT, tree.right is not None])
    if tree.right is not None:
        tree_array.extend(__tree_to_tree_array(tree.right))

    return tree_array


def __tree_array_to_numpy_array(tree_array: []) -> np.array:
    number_array = []

    for state, value in tree_array:

        if isinstance(value, bool):
            number_value = 1 if value else 0
        elif isinstance(value, int):
            number_value = value
        else:
            raise ValueError("Can't convert type to number: %s" % value)

        number_array.append([state.value, number_value])

    return np.array(number_array)


def __get_prime_factor_tree(x: int) -> PrimeFactorTree:
    def get_pairs(xs):
        for i in range(0, len(xs), 2):
            yield xs[i:i + 2]

    prime_factors = __get_prime_factors(x)

    current_nodes = [PrimeFactorTree(p, None, None) for p in prime_factors]

    while len(current_nodes) != 1:
        pairs = get_pairs(current_nodes)
        new_nodes = []

        for pair in pairs:
            if len(pair) == 2:
                new_nodes.append(PrimeFactorTree(pair[0].value * pair[1].value, pair[0], pair[1]))
            if len(pair) == 1:
                new_nodes.append(pair[0])

        current_nodes = new_nodes

    return current_nodes[0]


def __get_prime_factors(x: int) -> [int]:
    prime_factors = []

    i = 2
    while i <= x:
        if x % i == 0:  # If i is a factor of x
            prime_factors.append(i)
            x /= i
        else:
            i += 1

    return prime_factors
