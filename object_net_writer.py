import tensorflow as tf
import numpy as np
from typing import Callable


class ObjectNetWriter:

    GetNextStateFn = Callable[[np.array, float], np.array]

    def __init__(
            self,
            truth_states: tf.TensorArray,
            truth_outputs: tf.TensorArray,
            hidden_vector_size: int,
            batch_size: int,
            get_next_state_fn: GetNextStateFn):

        self.hidden_vector_size = hidden_vector_size
        self.batch_size = batch_size
        self.weights = [
            tf.Variable(tf.random_normal([self.hidden_vector_size, self.hidden_vector_size + 1])) for _ in range(4)]
        self.biases = [
            tf.Variable(tf.random_normal([self.hidden_vector_size + 1])) for _ in range(4)]
        self.get_next_state_fn = ObjectNetWriter.__wrap_state_function(get_next_state_fn)

        def batch_while_loop(step, batch_input_ta: tf.TensorArray, batch_output_ta: tf.TensorArray):
            current_input = batch_input_ta.read(step)
            current_hidden_vector = tf.zeros([self.hidden_vector_size], name="hidden_vector")
            current_output = tf.zeros([0], name="generated_outputs")

            # TODO: Investigate using TensorArrays for this while loop
            _, _, current_output = tf.while_loop(
                cond=ObjectNetWriter.__while_condition,
                body=self.__while_loop,
                loop_vars=[current_hidden_vector, current_input, current_output],
                shape_invariants=[
                    tf.TensorShape([self.hidden_vector_size]),  # `hidden_vector` has fixed length
                    tf.TensorShape([None]),  # `truth_states` has variable length
                    tf.TensorShape([None])])  # `object_outputs` has variable length

            batch_output_ta = batch_output_ta.write(step, current_output)

            return step + 1, batch_input_ta, batch_output_ta

        def batch_while_condition(step, batch_input_ta: tf.TensorArray, _):
            return step < batch_input_ta.size()

        _, _, self.generated_outputs_ta = tf.while_loop(
            cond=batch_while_condition,
            body=batch_while_loop,
            loop_vars=[
                tf.constant(0),
                truth_states,
                tf.TensorArray(dtype=tf.float32, size=self.batch_size)])

        self.cost = self.__get_cost(truth_outputs, self.generated_outputs_ta)

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

    def __get_cost(self, truth_outputs_ta: tf.TensorArray, generated_outputs_ta: tf.TensorArray):
        def condition(step, ta, *_):
            return step < ta.size()

        def loop(
                step,
                _truth_outputs_ta: tf.TensorArray,
                _generated_outputs_ta: tf.TensorArray,
                _costs_ta: tf.TensorArray):

            current_truth = _truth_outputs_ta.read(step)
            current_generated = _generated_outputs_ta.read(step)
            current_cost = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(current_truth - current_generated))))

            return step + 1, _truth_outputs_ta, _generated_outputs_ta, _costs_ta.write(step, current_cost)

        _, _, _, costs_ta = tf.while_loop(
            cond=condition,
            body=loop,
            loop_vars=[
                0,
                truth_outputs_ta,
                generated_outputs_ta,
                tf.TensorArray(dtype=tf.float32, size=self.batch_size)])

        return tf.reduce_mean(costs_ta.stack())

    @staticmethod
    def __wrap_state_function(get_next_state_fn: GetNextStateFn) -> GetNextStateFn:
        def wrapped(current_state: [int], current_value: [float]) -> [int]:
            return np.array([int(state.value) for state in get_next_state_fn(current_state, current_value)])

        return wrapped
