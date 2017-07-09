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


class PrimeFactorTreeState(Enum):
    VALUE = 0
    IS_EVEN = 1
    LEFT_OPT = 2
    RIGHT_OPT = 3


def get_trees(n: int) -> [PrimeFactorTree]:
    return [__get_prime_factor_tree(x) for x in range(2, n + 2)]


def get_tree_arrays(n: int) -> [np.array]:
    return [__tree_to_array(tree) for tree in get_trees(n)]


def get_next_state(current_state: [PrimeFactorTreeState], current_choice: float) -> [PrimeFactorTreeState]:
    if len(current_state) == 0:
        return [PrimeFactorTreeState.VALUE]

    state, *stacked_states = current_state

    next_state = []

    if state == PrimeFactorTreeState.VALUE:
        next_state = [PrimeFactorTreeState.IS_EVEN]
    elif state == PrimeFactorTreeState.IS_EVEN:
        next_state = [PrimeFactorTreeState.LEFT_OPT]
    elif state == PrimeFactorTreeState.LEFT_OPT:
        if current_choice > 0.5:
            next_state = [PrimeFactorTreeState.VALUE, PrimeFactorTreeState.RIGHT_OPT]
        else:
            next_state = [PrimeFactorTreeState.RIGHT_OPT]
    elif state == PrimeFactorTreeState.RIGHT_OPT:
        if current_choice > 0.5:
            next_state = [PrimeFactorTreeState.VALUE]
        else:
            next_state = []

    return next_state + stacked_states


def __tree_to_array(tree: PrimeFactorTree) -> [(int, [int])]:
    array = []

    array.append((PrimeFactorTreeState.VALUE.value, __outputs_to_numbers([tree.value])))
    array.append((PrimeFactorTreeState.IS_EVEN.value, __outputs_to_numbers([tree.is_even])))

    array.append((PrimeFactorTreeState.LEFT_OPT.value, __outputs_to_numbers([tree.left is not None])))
    if tree.left is not None:
        array.extend(__tree_to_array(tree.left))

    array.append((PrimeFactorTreeState.RIGHT_OPT.value, __outputs_to_numbers([tree.right is not None])))
    if tree.right is not None:
        array.extend(__tree_to_array(tree.right))

    return array


def __outputs_to_numbers(outputs: []) -> [float]:
    def convert(output) -> float:
        if isinstance(output, bool):
            return 1.0 if output else 0.0
        elif isinstance(output, int):
            return float(output)
        elif isinstance(output, float):
            return output
        else:
            raise ValueError("Can't convert type to number: %s" % output)

    return [convert(output) for output in outputs]


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
