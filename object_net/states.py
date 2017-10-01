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

    NONE = 3
    """Do not output anything at this state"""


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

    def __str__(self):
        return "State(%s, %d)" % (self.name, self.id)
