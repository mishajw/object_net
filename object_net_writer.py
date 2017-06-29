import tensorflow as tf
import numpy as np
from typing import Callable


class ObjectNetWriter:

    GetNextStateFn = Callable[[np.array, float], np.array]

    def __init__(
            self,
            truth_states: tf.Tensor,
            truth_outputs: tf.Tensor,
            hidden_vector_size: int,
            get_next_state_fn: GetNextStateFn):

        self.hidden_vector_size = hidden_vector_size
        self.weights = [
            tf.Variable(tf.random_normal([self.hidden_vector_size, self.hidden_vector_size + 1])) for _ in range(4)]
        self.biases = [
            tf.Variable(tf.random_normal([self.hidden_vector_size + 1])) for _ in range(4)]
        self.get_next_state_fn = ObjectNetWriter.__wrap_state_function(get_next_state_fn)

        hidden_vector_temp = tf.zeros([self.hidden_vector_size], name="hidden_vector")
        generated_outputs_temp = tf.zeros([0], name="generated_outputs")

        _, _, self.generated_outputs = tf.while_loop(
            cond=ObjectNetWriter.__while_condition,
            body=self.__while_loop,
            loop_vars=[hidden_vector_temp, truth_states, generated_outputs_temp],
            shape_invariants=[
                tf.TensorShape([self.hidden_vector_size]),  # `hidden_vector` has fixed length
                tf.TensorShape([None]),  # `truth_states` has variable length
                tf.TensorShape([None])])  # `object_outputs` has variable length

        self.cost = self.__get_cost(truth_outputs, self.generated_outputs)

    def __while_loop(self, hidden_vector, truth_states, object_outputs):
        current_state = truth_states[0]

        weights = tf.case(
            pred_fn_pairs=[
                (tf.equal(current_state, 0), lambda: self.weights[0]),
                (tf.equal(current_state, 1), lambda: self.weights[1]),
                (tf.equal(current_state, 2), lambda: self.weights[2]),
                (tf.equal(current_state, 3), lambda: self.weights[3])],
            default=lambda: self.weights[0],
            exclusive=True)

        biases = tf.case(
            pred_fn_pairs=[
                (tf.equal(current_state, 0), lambda: self.biases[0]),
                (tf.equal(current_state, 1), lambda: self.biases[1]),
                (tf.equal(current_state, 2), lambda: self.biases[2]),
                (tf.equal(current_state, 3), lambda: self.biases[3])],
            default=lambda: self.biases[0],
            exclusive=True)

        weights = tf.reshape(weights, [self.hidden_vector_size, self.hidden_vector_size + 1])
        biases = tf.reshape(biases, [self.hidden_vector_size + 1])

        activations = tf.squeeze(tf.matmul(tf.expand_dims(hidden_vector, axis=0), weights) + biases)

        next_hidden_vector = tf.sigmoid(tf.slice(activations, [0], [self.hidden_vector_size]))
        current_choice = tf.slice(activations, [self.hidden_vector_size], [-1])

        return next_hidden_vector, truth_states[1:], tf.concat([object_outputs, current_choice], axis=0)

    @staticmethod
    def __while_condition(_, truth_states, __):
        return tf.not_equal(tf.shape(truth_states)[0], 0)

    @staticmethod
    def __get_cost(truth_outputs: tf.Tensor, generated_outputs: tf.Tensor):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.abs(truth_outputs - generated_outputs))))

    @staticmethod
    def __wrap_state_function(get_next_state_fn: GetNextStateFn) -> GetNextStateFn:
        def wrapped(current_state: [int], current_value: [float]) -> [int]:
            return np.array([int(state.value) for state in get_next_state_fn(current_state, current_value)])

        return wrapped
