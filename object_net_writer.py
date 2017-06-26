import tensorflow as tf
import numpy as np
from typing import Callable, List


class ObjectNetWriter:

    GetNextStateFn = Callable[[np.array, float], np.array]

    def __init__(self, object_tensor: tf.Tensor, hidden_vector_size: int, get_next_state_fn: GetNextStateFn):
        hidden_vector = tf.zeros([hidden_vector_size], name="hidden_vector")

        get_next_state_fn = ObjectNetWriter.wrap_state_function(get_next_state_fn)

        states = get_next_state_fn([], 0)

        self.output_vector, self.output_states = tf.while_loop(
            cond=ObjectNetWriter.check_states_for_end,
            body=ObjectNetWriter.loop,
            loop_vars=[hidden_vector, states],
            shape_invariants=[tf.TensorShape([hidden_vector_size]), tf.TensorShape([None])])

    @staticmethod
    def loop(hidden_vector, states):
        return hidden_vector, states

    @staticmethod
    def check_states_for_end(_, states):
        return tf.not_equal(tf.shape(states)[0], 0)

    @staticmethod
    def wrap_state_function(get_next_state_fn: GetNextStateFn) -> GetNextStateFn:
        def wrapped(current_state: [int], current_value: [float]) -> [int]:
            return np.array([int(state.value) for state in get_next_state_fn(current_state, current_value)])

        return wrapped
