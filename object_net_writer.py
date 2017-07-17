from typing import Callable
import numpy as np
import tensorflow as tf
import tf_utils


class ObjectNetWriter:

    GetNextStateFn = Callable[[np.array, float], np.array]

    def __init__(
            self,
            truth_padded_data,
            initial_hidden_vector_input: tf.Tensor,
            hidden_vector_size: int,
            fully_connected_sizes: [int],
            state_outputs: [int],
            get_next_state_fn: GetNextStateFn):
        self.hidden_vector_size = hidden_vector_size
        self.fully_connected_sizes = fully_connected_sizes
        self.state_outputs = state_outputs
        self.get_next_state_fn = ObjectNetWriter.__wrap_state_function(get_next_state_fn)

        num_batches = truth_padded_data.batch_size

        def batch_while_loop(step, batch_output_ta: tf.TensorArray):
            current_step_count = truth_padded_data.step_counts[step]
            current_states_padded = truth_padded_data.states_padded[step]
            current_initial_hidden_vector_input = initial_hidden_vector_input[step]

            with tf.variable_scope("initial_hidden_vector"):
                current_hidden_vector = tf.sigmoid(self.__create_fully_connected_layers(
                    current_initial_hidden_vector_input, self.fully_connected_sizes + [self.hidden_vector_size]))

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
                ObjectNetWriter.__pad_ta_elements(
                    current_output, tf.shape(truth_padded_data.outputs_padded)[2]).stack())

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
            tf.shape(truth_padded_data.outputs_counts)[1]).stack()

        self.cost = self.__get_cost(truth_padded_data.outputs_padded, self.generated_outputs_padded)

    def __while_loop(
            self,
            step: int,
            current_step_count: tf.Tensor,
            current_states_padded: tf.Tensor,
            current_hidden_vector: tf.Tensor,
            current_output_ta: tf.TensorArray):

        current_state = current_states_padded[step]

        next_hidden_vector, current_choice = tf.case(
            pred_fn_pairs=[(
                tf.equal(current_state, i),
                lambda i=i: self.__create_guess_layers(current_hidden_vector, i))
                for i in range(len(self.state_outputs))],
            default=lambda: (current_hidden_vector, tf.constant(0, dtype=tf.float32) / tf.constant(0, tf.float32)))

        next_hidden_vector = tf.reshape(next_hidden_vector, [self.hidden_vector_size])

        return \
            step + 1, \
            current_step_count, \
            current_states_padded, \
            next_hidden_vector, \
            current_output_ta.write(step, current_choice)

    @staticmethod
    def __while_condition(step, current_step_count, *_):
        return step < current_step_count

    def __create_guess_layers(self, hidden_vector: tf.Tensor, state_number: int):
        with tf.variable_scope("guess_layers_%d" % state_number):
            num_outputs = self.state_outputs[state_number]

            with tf.variable_scope("lstm"):
                c, h = tf.split(hidden_vector, 2, axis=0)

                weights_lstm = tf.get_variable(
                    name="weights",
                    shape=[self.hidden_vector_size / 2, self.hidden_vector_size * 2],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer())

                biases_lstm = tf.get_variable(
                    name="biases",
                    shape=[self.hidden_vector_size * 2],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer())

                activations = tf.squeeze(tf.matmul(tf.expand_dims(h, axis=0), weights_lstm) + biases_lstm)

                input_gate, new_input, forget_gate, output_gate = tf.split(
                    value=activations, num_or_size_splits=4, axis=0)

                input_gate = tf.sigmoid(input_gate)
                new_input = tf.tanh(new_input)
                forget_gate = tf.sigmoid(forget_gate)
                output_gate = tf.sigmoid(output_gate)

                new_c = tf.multiply(forget_gate, c) + tf.multiply(input_gate, new_input)
                new_h = tf.multiply(output_gate, tf.tanh(c))

                next_hidden_vector = tf.concat([new_c, new_h], axis=0)

            weights = tf.get_variable(
                name="weights",
                shape=[self.hidden_vector_size / 2, num_outputs],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer())

            biases = tf.get_variable(
                name="biases",
                shape=[num_outputs],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

            current_choice = tf.squeeze(tf.matmul(tf.expand_dims(new_h, axis=0), weights) + biases, axis=0)

            return next_hidden_vector, current_choice

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

    @staticmethod
    def __create_fully_connected_layers(initial_input: tf.Tensor, sizes: [int]):
        current_input = initial_input
        current_input_size = initial_input.shape[0]

        for i, size in enumerate(sizes):
            weights = tf_utils.try_create_scoped_variable(
                "weights_%d" % i,
                shape=[current_input_size, size],
                initializer=tf.random_normal_initializer())

            biases = tf_utils.try_create_scoped_variable(
                "biases_%d" % i,
                shape=[size],
                initializer=tf.zeros_initializer())

            current_input = tf.squeeze(tf.matmul(tf.expand_dims(current_input, axis=0), weights) + biases)
            current_input_size = size

            if i < len(sizes) - 1:
                current_input = tf.sigmoid(current_input)

        return current_input
