from . import types
from typing import Any
import math


def add_arguments(parser):
    parser.add_argument("--num_data", type=int, default=10000, help="Amount of examples to load")


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


def log_normalise_tree(tree: Any):
    tree["value"] = math.log(tree["value"]) if tree["value"] > 0 else -1

    if tree["left"] is not None:
        log_normalise_tree(tree["left"])

    if tree["right"] is not None:
        log_normalise_tree(tree["right"])


def log_unnormalise_tree(tree: Any):
    tree["value"] = math.pow(math.e, tree["value"]) if tree["value"] > 0 else -1

    if tree["left"] is not None:
        log_unnormalise_tree(tree["left"])

    if tree["right"] is not None:
        log_unnormalise_tree(tree["right"])


def create_tree(value: int, left: Any, right: Any):
    mod_three_int = int(value) % 3
    if mod_three_int == 0:
        mod_three = "zero"
    elif mod_three_int == 1:
        mod_three = "one"
    elif mod_three_int == 2:
        mod_three = "two"
    else:
        raise ValueError()

    return {
        "value": value,
        "mod_three": mod_three,
        "left": left,
        "right": right
    }


def get_trees(args) -> [Any]:
    return [__get_prime_factor_tree(x) for x in range(2, args.num_data + 2)]


def __get_prime_factor_tree(x: int) -> Any:
    def get_pairs(xs):
        for i in range(0, len(xs), 2):
            yield xs[i:i + 2]

    prime_factors = __get_prime_factors(x)

    current_nodes = [create_tree(p, None, None) for p in prime_factors]

    while len(current_nodes) != 1:
        pairs = get_pairs(current_nodes)
        new_nodes = []

        for pair in pairs:
            if len(pair) == 2:
                new_nodes.append(create_tree(pair[0]["value"] * pair[1]["value"], pair[0], pair[1]))
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
