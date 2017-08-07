import itertools
import math
import numpy as np
import state_transition_type
import states
import types


def add_arguments(parser):
    parser.add_argument("--num_data", type=int, default=10000, help="Amount of examples to load")
    parser.add_argument("--normalize_factor", type=int, default=None)
    parser.add_argument("--log_normalize", type=bool, default=False)


state_encoder = states.StateEncoder(["value", "mod_three", "left_opt", "right_opt"])


class PrimeFactorTree:
    def __init__(self, value: float, left, right):
        self.value = value
        self.left = left
        self.right = right

        self.mod_three = [0, 0, 0]
        self.mod_three[int(self.value) % 3] = 1

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
        self.value = math.pow(self.value, math.e) if self.value > 0 else -1

        if self.left is not None and self.right is not None:
            self.left.pow_e()
            self.right.pow_e()


def get_trees(args) -> [PrimeFactorTree]:
    trees = [__get_prime_factor_tree(x) for x in range(2, args.num_data + 2)]

    if args.normalize_factor is not None:
        for tree in trees:
            tree.multiply(1 / args.normalize_factor)

    if args.log_normalize:
        for tree in trees:
            tree.log()

    return trees


def get_next_state(state_stack: [int], output: np.array) -> state_transition_type.StateTransitionType:
    state, state_stack = state_stack

    state_string = state_encoder.decode(state)

    if state_string is "value":
        return state_transition_type.ContinueObject(state_encoder.encode("mod_three"))
    elif state_string is "mod_three":
        return state_transition_type.ContinueObject(state_encoder.encode("left_opt"))
    elif state_string is "left_opt":
        if output[0] > 0.5:
            return state_transition_type.StartObject(state_encoder.encode("value"), state_encoder.encode("right_opt"))
        else:
            return state_transition_type.ContinueObject(state_encoder.encode("right_opt"))
    elif state_string is "right_opt":
        if output[0] > 0.5:
            return state_transition_type.StartObjectThenStop(state_encoder.encode("value"))
        else:
            return state_transition_type.StopObject()


def tree_to_array(tree: PrimeFactorTree, args) -> [(int, [int])]:
    array = []

    array.append((state_encoder.encode("value"), __outputs_to_numbers([tree.value])))
    array.append((state_encoder.encode("mod_three"), __outputs_to_numbers(tree.mod_three)))

    array.append((state_encoder.encode("left_opt"), __outputs_to_numbers([tree.left is not None])))
    if tree.left is not None:
        array.extend(tree_to_array(tree.left, args))

    array.append((state_encoder.encode("right_opt"), __outputs_to_numbers([tree.right is not None])))
    if tree.right is not None:
        array.extend(tree_to_array(tree.right, args))

    return array


def array_to_tree(initial_array: [(int, [int])], args) -> PrimeFactorTree:
    def get_subtree(_array, choice_state: int) -> (PrimeFactorTree, [(int, [int])]):
        _state, _outputs = next(_array)
        assert _state == choice_state
        assert len(_outputs) == 1

        # Peek at what's next in `array`, but place back in
        next_state, next_outputs = next(_array)
        _array = itertools.chain([(next_state, next_outputs)], _array)

        if next_state == state_encoder.encode("value"):
            return get_tree(_array)
        else:
            return None, _array

    def get_tree(array: [(int, [int])]) -> (PrimeFactorTree, [(int, [int])]):
        try:
            state, outputs = next(array)
        except StopIteration:
            return None

        assert state == state_encoder.encode("value")
        assert len(outputs) == 1
        value = outputs[0]

        state, outputs = next(array)
        assert state == state_encoder.encode("mod_three")
        assert len(outputs) == 3
        mod_three = outputs

        left, array = get_subtree(array, state_encoder.encode("left_opt"))

        try:
            right, array = get_subtree(array, state_encoder.encode("right_opt"))
        except StopIteration:
            right = None

        final_tree = PrimeFactorTree(value, left, right)
        final_tree.mod_three = mod_three

        return final_tree, array

    # Ensure that `array` is a generator
    if not isinstance(initial_array, types.GeneratorType):
        initial_array = iter(initial_array)

    tree, _ = get_tree(initial_array)

    if args.normalize_factor is not None:
        tree.multiply(args.normalize_factor)

    if args.log_normalize:
        tree.pow_e()

    return tree


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
