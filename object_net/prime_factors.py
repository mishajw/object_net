from . import states
from . import types
import math


def add_arguments(parser):
    parser.add_argument("--num_data", type=int, default=10000, help="Amount of examples to load")
    parser.add_argument("--normalize_factor", type=int, default=None)
    parser.add_argument("--log_normalize", type=bool, default=False)


state_encoder = states.StateEncoder(["value", "mod_three", "left_opt", "right_opt"])


def get_prime_factor_tree_type():
    all_types = types.create_from_dict(
        {
            "types": [
                {
                    "base": "object",
                    "name": "tree",
                    "value": "int",
                    "mod_three": "mod_three",
                    "left": "optional[tree]",
                    "right": "optional[tree]"
                },
                {
                    "base": "enum",
                    "name": "mod_three",
                    "options": ["zero", "one", "two"]
                },
                {
                    "base": "optional",
                    "type": "tree"
                }
            ]
        })

    return all_types[0]


class PrimeFactorTree:
    def __init__(self, value: float, left, right):
        self.value = value
        self.left = left
        self.right = right

        mod_three_int = int(self.value) % 3
        if mod_three_int == 0:
            self.mod_three = "zero"
        elif mod_three_int == 1:
            self.mod_three = "one"
        elif mod_three_int == 2:
            self.mod_three = "two"

    def __str__(self):
        if self.left is None and self.right is None:
            return "%.1f" % self.value
        else:
            return "(%.1f = %s * %s)" % (self.value, self.left, self.right)

    def multiply(self, x):
        self.value *= x

        if self.left is not None and self.right is not None:
            self.left.multiply(x)
            self.right.multiply(x)

    def log(self):
        self.value = math.log(self.value) if self.value > 0 else -1

        if self.left is not None and self.right is not None:
            self.left.log()
            self.right.log()

    def pow_e(self):
        self.value = math.pow(math.e, self.value) if self.value > 0 else -1

        if self.left is not None and self.right is not None:
            self.left.pow_e()
            self.right.pow_e()

    def to_dict(self) -> dict:
        value = {
            "value": self.value,
            "mod_three": self.mod_three,
            "left": {},
            "right": {}
        }

        if self.left is not None:
            value["left"] = self.left.to_dict()

        if self.right is not None:
            value["right"] = self.right.to_dict()

        return value


def get_trees(args) -> [dict]:
    trees = [__get_prime_factor_tree(x) for x in range(2, args.num_data + 2)]

    if args.normalize_factor is not None:
        for tree in trees:
            tree.multiply(1 / args.normalize_factor)

    if args.log_normalize:
        for tree in trees:
            tree.log()

    return [tree.to_dict() for tree in trees]


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
