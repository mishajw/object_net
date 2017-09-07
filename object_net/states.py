from . import state_stack
from enum import Enum
from typing import Callable, Tuple, List
import tensorflow as tf


UpdateStateStackFn = Callable[[state_stack.StateStack, tf.Tensor, tf.Tensor], Tuple[state_stack.StateStack, tf.Tensor]]
"""
Type for functions that give transitions between states by modifying a state stack
The function takes the current state stack, the current hidden vector, and the output from the current choice
The function returns the new state stack, and optionally the return value of the child created
"""


class OutputType(Enum):
    BOOL = 0
    """Output in the range 0 to 1 where `< 0.5` is false and `>= 0.5` is true"""

    SIGNED = 1
    """Output in the range -1 to 1"""

    REAL = 2
    """Output in the range negative infinity to infinity"""


class State:
    def __init__(self, name: str, num_outputs: int, output_type: OutputType):
        self.name = name
        self.num_outputs = num_outputs
        self.output_type = output_type
        self.id = None

    @staticmethod
    def assign_ids(states: List["State"]):
        for i, state in enumerate(states):
            state.id = i


class StateEncoder:
    """Encode states strings into integers for NumPy arrays"""

    end_state_string = "end"

    def __init__(self, state_strings: [str]):
        self.state_strings = state_strings

    def encode(self, state_string: str) -> int:
        assert state_string in self.state_strings

        if state_string == StateEncoder.end_state_string:
            return 0

        # Plus one because 0 is taken up by end state
        return self.state_strings.index(state_string) + 1

    def get_end_state(self):
        return self.encode(StateEncoder.end_state_string)

    def decode(self, state_int: int):
        assert state_int < len(self.state_strings) + 1

        if state_int == 0:
            return StateEncoder.end_state_string

        return self.state_strings[state_int + 1]
