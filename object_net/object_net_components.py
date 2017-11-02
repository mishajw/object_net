from . import states
from typing import List
import tensorflow as tf
import tf_utils


class HiddenVectorCombiner:
    def __call__(
            self,
            parent_hidden_vector: tf.Tensor,
            child_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()


class AdditionHiddenVectorCombiner(HiddenVectorCombiner):
    def __call__(
            self,
            parent_hidden_vector: tf.Tensor,
            child_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor) -> tf.Tensor:

        all_hidden_vectors = tf.cond(
            tf.reduce_any(tf.is_nan(child_hidden_vector)),
            lambda: tf.concat([[parent_hidden_vector], [inner_hidden_vector]], axis=0),
            lambda: tf.concat([[parent_hidden_vector], [child_hidden_vector], [inner_hidden_vector]], axis=0))

        return tf.reduce_sum(all_hidden_vectors, axis=0)


class MostRecentHiddenVectorCombiner(HiddenVectorCombiner):
    def __call__(
            self,
            parent_hidden_vector: tf.Tensor,
            child_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor) -> tf.Tensor:

        # If `child_hidden_vector` is not NaNs, then it is the most recent. Otherwise, `child_hidden_vector` is the most
        # recent hidden vector
        return tf.cond(
            tf.reduce_any(tf.is_nan(child_hidden_vector)),
            lambda: inner_hidden_vector,
            lambda: child_hidden_vector)


class HiddenVectorNetwork:
    def __call__(
            self,
            parent_hidden_vector: tf.Tensor,
            child_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor,
            state: states.State) -> (tf.Tensor, tf.Tensor):
        raise NotImplementedError()

    def get_hidden_vector_size(self) -> int:
        raise NotImplementedError()


class FullyConnectedHiddenVectorNetwork(HiddenVectorNetwork):
    def __init__(self, layer_sizes: List[int], hidden_vector_size: int, hidden_vector_combiner: HiddenVectorCombiner):
        self.layer_sizes = layer_sizes
        self.hidden_vector_size = hidden_vector_size
        self.hidden_vector_combiner = hidden_vector_combiner

    def __call__(
            self, parent_hidden_vector: tf.Tensor,
            child_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor,
            state: states.State) -> (tf.Tensor, tf.Tensor):

        with tf.variable_scope("make_choice_%s" % tf_utils.format_for_scope(state.name)):
            hidden_vector = self.hidden_vector_combiner(parent_hidden_vector, child_hidden_vector, inner_hidden_vector)

            final_activations = tf.squeeze(tf_utils.get_fully_connected_layers(
                tf.expand_dims(hidden_vector, axis=0),
                self.layer_sizes + [self.hidden_vector_size + state.num_outputs], tf.tanh), axis=0)

            next_hidden_vector, output = tf.split(final_activations, [self.hidden_vector_size, state.num_outputs])

            output = state.format_tensor(output)

        return next_hidden_vector, output

    def get_hidden_vector_size(self) -> int:
        return self.hidden_vector_size


class LstmHiddenVectorNetwork(HiddenVectorNetwork):
    def __init__(self, layer_size: int, num_layers: int, hidden_vector_combiner: HiddenVectorCombiner):
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.hidden_vector_combiner = hidden_vector_combiner

    def __call__(
            self,
            parent_hidden_vector: tf.Tensor,
            child_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor,
            state: states.State) -> (tf.Tensor, tf.Tensor):

        total_hidden_vector = self.hidden_vector_combiner(
            parent_hidden_vector, child_hidden_vector, inner_hidden_vector)

        layer_hidden_vectors = tf.split(total_hidden_vector, self.num_layers)

        current_input = tf.zeros(self.layer_size / 2)

        next_hidden_vector_pieces = []

        for i in range(self.num_layers):
            with tf.variable_scope("lstm_layer_%d" % i):
                c, h = tf.split(layer_hidden_vectors[i], 2, axis=0)
                new_c, new_h = self.__lstm(current_input, c, h)
                current_input = new_h
                next_hidden_vector_pieces.extend([new_c, new_h])

            # next_hidden_vector = tf.concat([new_c, new_h], axis=0)

        with tf.variable_scope("make_choice_%s" % tf_utils.format_for_scope(state.name)):
            weights = tf_utils.try_create_scoped_variable(
                name="weights",
                shape=[self.layer_size / 2, state.num_outputs],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer())

            biases = tf_utils.try_create_scoped_variable(
                name="biases",
                shape=[state.num_outputs],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

            current_choice = tf.squeeze(tf.matmul(tf.expand_dims(current_input, axis=0), weights) + biases, axis=0)

        current_choice = state.format_tensor(current_choice)

        next_hidden_vector = tf.concat(next_hidden_vector_pieces, axis=0)

        return next_hidden_vector, current_choice

    def get_hidden_vector_size(self) -> int:
        return self.layer_size * self.num_layers

    def __lstm(self, layer_input: tf.Tensor, h_in: tf.Tensor, c_in: tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        weights_lstm = tf_utils.try_create_scoped_variable(
            name="weights",
            shape=[self.layer_size, self.layer_size * 2],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer())

        biases_lstm = tf_utils.try_create_scoped_variable(
            name="biases",
            shape=[self.layer_size * 2],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())

        activations = tf.squeeze(
            tf.matmul(tf.expand_dims(tf.concat([h_in, layer_input], axis=0), axis=0), weights_lstm) + biases_lstm)

        input_gate, new_input, forget_gate, output_gate = tf.split(
            value=activations, num_or_size_splits=4, axis=0)

        input_gate = tf.sigmoid(input_gate)
        new_input = tf.tanh(new_input)
        forget_gate = tf.sigmoid(forget_gate)
        output_gate = tf.sigmoid(output_gate)

        new_c = tf.multiply(forget_gate, c_in) + tf.multiply(input_gate, new_input)
        new_h = tf.multiply(output_gate, tf.tanh(c_in))

        return new_c, new_h
