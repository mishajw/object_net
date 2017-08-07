from typing import Callable
import numpy as np
import state_transition_type


GetNextStateFn = Callable[[int, np.array], state_transition_type.StateTransitionType]
"""
Type for functions that give transitions between states
The function takes the current state as an integer, and the output from the current choice
The function returns the next state as an integer, and the `StateTransitionType`
"""


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
