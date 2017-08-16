from . import states
import tensorflow as tf
import tf_utils


class InnerHiddenVectorCreator:
    def __call__(
            self,
            parent_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor,
            num_outputs: int) -> (tf.Tensor, tf.Tensor):
        pass


class LstmInnerHiddenVectorCreator(InnerHiddenVectorCreator):
    def __init__(self, hidden_vector_size: int):
        self.hidden_vector_size = hidden_vector_size

    def __call__(
            self,
            parent_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor,
            states_output_description: states.OutputDescription) -> (tf.Tensor, tf.Tensor):

        hidden_vector = tf.add(inner_hidden_vector, parent_hidden_vector) / 2

        with tf.variable_scope("lstm"):
            c, h = tf.split(hidden_vector, 2, axis=0)

            weights_lstm = tf_utils.try_create_scoped_variable(
                name="weights",
                shape=[self.hidden_vector_size / 2, self.hidden_vector_size * 2],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer())

            biases_lstm = tf_utils.try_create_scoped_variable(
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

        weights = tf_utils.try_create_scoped_variable(
            name="weights",
            shape=[self.hidden_vector_size / 2, states_output_description.num_outputs],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer())

        biases = tf_utils.try_create_scoped_variable(
            name="biases",
            shape=[states_output_description.num_outputs],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())

        current_choice = tf.squeeze(tf.matmul(tf.expand_dims(new_h, axis=0), weights) + biases, axis=0)

        if states_output_description.output_type == states.OutputType.BOOL:
            current_choice = tf.sigmoid(current_choice)
        elif states_output_description.output_type == states.OutputType.SIGNED:
            current_choice = tf.tanh(current_choice)
        elif states_output_description.output_type == states.OutputType.REAL:
            # Output is already in the range of real numbers
            pass

        return next_hidden_vector, current_choice


class ChildHiddenVectorCombiner:
    def __call__(
            self,
            parent_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor,
            child_hidden_vector: tf.Tensor) -> tf.Tensor:
        pass


class AdditionChildHiddenVectorCombiner(ChildHiddenVectorCombiner):
    def __call__(
            self,
            parent_hidden_vector: tf.Tensor,
            inner_hidden_vector: tf.Tensor,
            child_hidden_vector: tf.Tensor) -> tf.Tensor:
        return tf.add(inner_hidden_vector, child_hidden_vector)
