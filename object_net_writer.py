from typing import Callable
import numpy as np
import tensorflow as tf


class ObjectNetWriter:

    GetNextStateFn = Callable[[np.array, float], np.array]

    def __init__(
            self,
            truth_step_counts: tf.Tensor,
            truth_outputs_counts: tf.Tensor,
            truth_states_padded: tf.Tensor,
            truth_outputs_padded: tf.Tensor,
            hidden_vector_size: int,
            get_next_state_fn: GetNextStateFn):

        self.hidden_vector_size = hidden_vector_size
        self.weights = [
            tf.Variable(tf.random_normal([self.hidden_vector_size, self.hidden_vector_size + 1]), name="weights_%d" % i)
            for i in range(4)]
        self.biases = [
            tf.Variable(tf.random_normal([self.hidden_vector_size + 1]), name="biases_%d" % i) for i in range(4)]
        self.get_next_state_fn = ObjectNetWriter.__wrap_state_function(get_next_state_fn)

        num_batches = tf.shape(truth_step_counts)[0]

        def batch_while_loop(step, batch_output_ta: tf.TensorArray):
            current_step_count = truth_step_counts[step]
            current_states_padded = truth_states_padded[step]

            current_hidden_vector = tf.zeros([self.hidden_vector_size], name="hidden_vector")
            current_output_ta = tf.TensorArray(dtype=tf.float32, size=current_step_count, name="current_output_ta")

            *_, current_output = tf.while_loop(
                cond=ObjectNetWriter.__while_condition,
                body=self.__while_loop,
                loop_vars=[
                    0,
                    current_step_count,
                    current_states_padded,
                    current_hidden_vector,
                    current_output_ta],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([self.hidden_vector_size]),
                    tf.TensorShape([])],
                name="step_while_loop")

            return step + 1, batch_output_ta.write(
                step,
                ObjectNetWriter.__pad_ta_elements(current_output, tf.shape(truth_outputs_padded)[2]).stack())

        def batch_while_condition(step, *_):
            return step < num_batches

        *_, generated_outputs_padded_ta = tf.while_loop(
            cond=batch_while_condition,
            body=batch_while_loop,
            loop_vars=[
                tf.constant(0),
                tf.TensorArray(dtype=tf.float32, size=num_batches)],
            name="batch_while_loop")

        self.generated_outputs_padded = ObjectNetWriter.__pad_ta_elements(
            generated_outputs_padded_ta,
            tf.shape(truth_outputs_counts)[1]).stack()

        self.cost = self.__get_cost(truth_outputs_padded, self.generated_outputs_padded)

    def __while_loop(
            self,
            step: int,
            current_step_count: tf.Tensor,
            current_states_padded: tf.Tensor,
            current_hidden_vector: tf.Tensor,
            current_output_ta: tf.TensorArray):

        current_state = current_states_padded[step]

        weights = tf.case(
            pred_fn_pairs=[
                (tf.equal(current_state, 0), lambda: self.weights[0]),
                (tf.equal(current_state, 1), lambda: self.weights[1]),
                (tf.equal(current_state, 2), lambda: self.weights[2]),
                (tf.equal(current_state, 3), lambda: self.weights[3])],
            default=lambda: self.weights[0],
            exclusive=True,
            name="selected_weights")

        biases = tf.case(
            pred_fn_pairs=[
                (tf.equal(current_state, 0), lambda: self.biases[0]),
                (tf.equal(current_state, 1), lambda: self.biases[1]),
                (tf.equal(current_state, 2), lambda: self.biases[2]),
                (tf.equal(current_state, 3), lambda: self.biases[3])],
            default=lambda: self.biases[0],
            exclusive=True,
            name="selected_biases")

        activations = tf.squeeze(tf.matmul(tf.expand_dims(current_hidden_vector, axis=0), weights) + biases)

        next_hidden_vector = tf.sigmoid(tf.slice(activations, [0], [self.hidden_vector_size]))
        current_choice = tf.slice(activations, [self.hidden_vector_size], [-1])

        return \
            step + 1, \
            current_step_count, \
            current_states_padded, \
            next_hidden_vector, \
            current_output_ta.write(step, current_choice)

    @staticmethod
    def __while_condition(step, current_step_count, *_):
        return step < current_step_count

    @staticmethod
    def __get_cost(truth_outputs_padded, generated_outputs_padded):
        with tf.variable_scope("cost"):
            tf.assert_equal(tf.shape(truth_outputs_padded), tf.shape(generated_outputs_padded))

            return tf.sqrt(
                tf.reduce_mean(tf.square(tf.abs(truth_outputs_padded - generated_outputs_padded))), name="cost")

    @staticmethod
    def __wrap_state_function(get_next_state_fn: GetNextStateFn) -> GetNextStateFn:
        def wrapped(current_state: [int], current_value: [float]) -> [int]:
            return np.array([int(state.value) for state in get_next_state_fn(current_state, current_value)])

        return wrapped

    @staticmethod
    def __pad_ta_elements(ta: tf.TensorArray, size: int) -> tf.TensorArray:
        with tf.variable_scope("pad_ta_elements"):
            def body(step, padded_ta):
                current = ta.read(step)
                padding_shape = tf.concat([[size - tf.shape(current)[0]], tf.shape(current)[1:]], axis=0)
                current_padded = tf.concat([current, tf.zeros(padding_shape, dtype=tf.float32)], axis=0)

                padded_ta = padded_ta.write(step, current_padded)

                return step + 1, padded_ta

            _, ta = tf.while_loop(
                cond=lambda step, _: step < ta.size(),
                body=body,
                loop_vars=[0, tf.TensorArray(dtype=tf.float32, size=ta.size())])

            return ta
