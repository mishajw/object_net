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

    def format_tensor(self, tensor: tf.Tensor):
        """
        Format a tensor to be in the correct ranges for the type of this state - for example, if the output type is
        boolean, the tensor will be formatted to be in the range 0-1
        :param tensor: the tensor to format
        :return: the formatted tensor of the same shape
        """
        if self.output_type == OutputType.BOOL:
            return tf.sigmoid(tensor)
        elif self.output_type == OutputType.SIGNED:
            return tf.tanh(tensor)
        elif self.output_type == OutputType.REAL:
            # Output is already in the range of real numbers
            return tensor
        elif self.output_type == OutputType.NONE:
            # Output does not exist therefore doesn't need editing
            return tensor
        else:
            raise ValueError("Output type is not recognised in state %s: %s" % (self, self.output_type))

    def __str__(self):
        return "State(%s, %d)" % (self.name, self.id)
