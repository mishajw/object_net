from typing import Callable
import numpy as np
import object_net_components
import padder
import tensorflow as tf
import tf_utils


class ObjectNetWriter:
    """
    Learn to write objects using regression using ObjectNet structures
    """

    GetNextStateFn = Callable[[np.array, float], np.array]
    """Type for functions that give transitions between states"""

    def __init__(
            self,
            truth_padded_data: padder.PlaceholderPaddedData,
            initial_hidden_vector_input: tf.Tensor,
            hidden_vector_size: int,
            fully_connected_sizes: [int],
            state_outputs: [int],
            get_next_state_fn: GetNextStateFn,
            inner_hidden_vector_creator: object_net_components.InnerHiddenVectorCreator,
            child_hidden_vector_combiner: object_net_components.ChildHiddenVectorCombiner):
        """
        Initialise TensorFlow graph
        :param truth_padded_data: the input data
        :param initial_hidden_vector_input: the inputs for each example in the batch
        :param hidden_vector_size: size of hidden vectors
        :param fully_connected_sizes: the sizes for fully connected layers
        :param state_outputs: the respective sizes of outputs for each state
        :param get_next_state_fn: function that gives transitions between states
        """
        self.hidden_vector_size = hidden_vector_size
        self.fully_connected_sizes = fully_connected_sizes
        self.state_outputs = state_outputs
        self.get_next_state_fn = ObjectNetWriter.__wrap_state_function(get_next_state_fn)
        self.inner_hidden_vector_creator = inner_hidden_vector_creator
        self.child_hidden_vector_combiner = child_hidden_vector_combiner

        num_batches = truth_padded_data.batch_size

        generated_outputs_padded_ta = self.__batch_while_loop(
            truth_padded_data, initial_hidden_vector_input, num_batches)

        self.generated_outputs_padded = ObjectNetWriter.__pad_ta_elements(
            generated_outputs_padded_ta,
            tf.shape(truth_padded_data.outputs_counts)[1]).stack()

        self.cost = self.__get_cost(truth_padded_data.outputs_padded, self.generated_outputs_padded)

    def __batch_while_loop(
            self,
            truth_padded_data: padder.PlaceholderPaddedData,
            initial_hidden_vector_input: tf.Tensor,
            num_batches: int) -> tf.TensorArray:

        def body(step, batch_output_ta: tf.TensorArray):
            current_step_count = truth_padded_data.step_counts[step]
            current_states_padded = truth_padded_data.states_padded[step]
            current_initial_hidden_vector_input = initial_hidden_vector_input[step]

            with tf.variable_scope("initial_hidden_vector"):
                current_hidden_vector = tf.sigmoid(self.__create_fully_connected_layers(
                    current_initial_hidden_vector_input, self.fully_connected_sizes + [self.hidden_vector_size]))

            current_output = self.__step_while_loop(current_step_count, current_states_padded, current_hidden_vector)

            return step + 1, batch_output_ta.write(
                step,
                ObjectNetWriter.__pad_ta_elements(
                    current_output, tf.shape(truth_padded_data.outputs_padded)[2]).stack())

        def cond(step, *_):
            return step < num_batches

        *_, generated_outputs_padded_ta = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                tf.constant(0),
                tf.TensorArray(dtype=tf.float32, size=num_batches)],
            name="batch_while_loop")

        return generated_outputs_padded_ta

    def __step_while_loop(self, step_count: int, states_padded: tf.Tensor, hidden_vector: tf.Tensor):
        def create_guess_layers(_hidden_vector: tf.Tensor, state_number: int):
            with tf.variable_scope("guess_layers_%d" % state_number):
                num_outputs = self.state_outputs[state_number]
                return self.inner_hidden_vector_creator(tf.zeros_like(_hidden_vector), _hidden_vector, num_outputs)

        def body(
                step: int,
                current_step_count: tf.Tensor,
                current_states_padded: tf.Tensor,
                current_hidden_vector: tf.Tensor,
                current_output_ta: tf.TensorArray):

            current_state = current_states_padded[step]

            next_hidden_vector, current_choice = tf.case(
                pred_fn_pairs=[(
                    tf.equal(current_state, i),
                    lambda i=i: create_guess_layers(current_hidden_vector, i))
                    for i in range(len(self.state_outputs))],
                default=lambda: (current_hidden_vector, tf.constant(0, dtype=tf.float32) / tf.constant(0, tf.float32)))

            next_hidden_vector = tf.reshape(next_hidden_vector, [self.hidden_vector_size])

            return \
                step + 1, \
                current_step_count, \
                current_states_padded, \
                next_hidden_vector, \
                current_output_ta.write(step, current_choice)

        def cond(step, current_step_count, *_):
            return step < current_step_count

        *_, current_output = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                0,
                step_count,
                states_padded,
                hidden_vector,
                tf.TensorArray(dtype=tf.float32, size=step_count, name="current_output_ta")],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([self.hidden_vector_size]),
                tf.TensorShape([])],
            name="step_while_loop")

        return current_output

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
