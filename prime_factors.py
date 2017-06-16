from enum import Enum


class PrimeFactorTree:
    def __init__(self, value: int, left, right):
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        if self.left is None and self.right is None:
            return str(self.value)
        else:
            return "(%d = %s * %s)" % (self.value, self.left, self.right)


class PrimeFactorTreeStates(Enum):
    VALUE = 0
    LEFT_OPT = 1
    RIGHT_OPT = 2


def get_examples() -> [PrimeFactorTree]:
    return [__get_prime_factor_tree(x) for x in range(3, 30)]


def tree_to_array(tree: PrimeFactorTree) -> []:
    tree_array = []

    tree_array.append([PrimeFactorTreeStates.VALUE, tree.value])

    tree_array.append([PrimeFactorTreeStates.LEFT_OPT, tree.left is not None])
    if tree.left is not None:
        tree_array.extend(tree_to_array(tree.left))

    tree_array.append([PrimeFactorTreeStates.RIGHT_OPT, tree.right is not None])
    if tree.right is not None:
        tree_array.extend(tree_to_array(tree.right))

    return tree_array


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
