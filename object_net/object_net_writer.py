from . import object_net_components
from . import padder
from . import state_stack
from . import state_transition
from . import types
import numpy as np
import tensorflow as tf
import tf_utils


class ObjectNetWriter:
    """
    Learn to write objects using regression using ObjectNet structures
    """

    def __init__(
            self,
            truth_padded_data: padder.PlaceholderPaddedData,
            initial_hidden_vector_input: tf.Tensor,
            object_type: types.Type,
            training: bool, # TODO: rename
            hidden_vector_network: object_net_components.HiddenVectorNetwork):
        """
        Initialise TensorFlow graph
        :param truth_padded_data: the input data
        :param initial_hidden_vector_input: the inputs for each example in the batch
        :param object_type: the object type we are writing
        :param training: true if training
        :param hidden_vector_network: the network to run when passing through hidden vectors
        """
        self.hidden_vector_size = hidden_vector_network.get_hidden_vector_size()
        self.object_type = object_type
        self.training = training
        self.hidden_vector_network = hidden_vector_network

        self.max_steps = tf.shape(truth_padded_data.states_padded)[1]

        num_batches = truth_padded_data.batch_size

        generated_states_padded_ta, \
            generated_outputs_padded_ta, \
            generated_outputs_counts_padded_ta, \
            generated_step_counts_ta = self.__batch_while_loop(
                truth_padded_data, initial_hidden_vector_input, num_batches)

        self.generated_states_padded = ObjectNetWriter.__pad_ta_elements(
            generated_states_padded_ta, self.max_steps).stack()

        self.generated_outputs_padded = ObjectNetWriter.__pad_ta_elements(
            generated_outputs_padded_ta, self.max_steps).stack()

        self.generated_outputs_counts_padded = ObjectNetWriter.__pad_ta_elements(
            generated_outputs_counts_padded_ta, self.max_steps).stack()

        self.generated_step_counts = generated_step_counts_ta.stack()

        self.cost = self.__get_cost(truth_padded_data.outputs_padded, self.generated_outputs_padded)

    def __batch_while_loop(
            self,
            truth_padded_data: padder.PlaceholderPaddedData,
            initial_hidden_vector_input: tf.Tensor,
            num_batches: int) -> (tf.TensorArray, tf.TensorArray):

        def body(
                step,
                batch_states_ta: tf.TensorArray,
                batch_outputs_ta: tf.TensorArray,
                batch_outputs_counts_ta: tf.TensorArray,
                batch_step_counts_ta: tf.TensorArray):

            with tf.variable_scope("initial_hidden_vector"):
                current_initial_hidden_vector_input = initial_hidden_vector_input[step]
                current_hidden_vector = tf.fill([self.hidden_vector_size], value=current_initial_hidden_vector_input)

            current_states_ta, current_outputs_ta, current_outputs_counts_ta, current_step_count = \
                self.__step_while_loop(
                    truth_padded_data.step_counts[step],
                    truth_padded_data.outputs_padded[step],
                    truth_padded_data.outputs_counts[step],
                    current_hidden_vector)

            return \
                step + 1, \
                batch_states_ta.write(step, current_states_ta.stack()), \
                batch_outputs_ta.write(
                    step,
                    ObjectNetWriter.__pad_ta_elements(
                        current_outputs_ta, tf.shape(truth_padded_data.outputs_padded)[2]).stack()), \
                batch_outputs_counts_ta.write(step, current_outputs_counts_ta.stack()), \
                batch_step_counts_ta.write(step, current_step_count)

        def cond(step, *_):
            return step < num_batches

        *_, \
            generated_states_padded_ta, \
            generated_outputs_padded_ta, \
            generated_outputs_counts_padded_ta, \
            generated_step_counts_ta = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[
                    tf.constant(0),
                    tf.TensorArray(dtype=tf.float32, size=num_batches),
                    tf.TensorArray(dtype=tf.float32, size=num_batches),
                    tf.TensorArray(dtype=tf.int32, size=num_batches),
                    tf.TensorArray(dtype=tf.int32, size=num_batches)],
                name="batch_while_loop")

        return \
            generated_states_padded_ta, \
            generated_outputs_padded_ta, \
            generated_outputs_counts_padded_ta, \
            generated_step_counts_ta

    def __step_while_loop(
            self,
            step_count: int,
            truth_outputs_padded: tf.Tensor,
            truth_outputs_counts: tf.Tensor,
            initial_hidden_vector: tf.Tensor):

        def create_guess_layers(
                parent_hidden_vector: tf.Tensor,
                child_hidden_vector: tf.Tensor,
                inner_hidden_vector: tf.Tensor,
                state_number: int):
            state = self.object_type.get_all_states()[state_number]
            return self.hidden_vector_network(
                parent_hidden_vector, child_hidden_vector, inner_hidden_vector, state)

        def body(
                step: int,
                stack_1,
                stack_2,
                states_ta: tf.TensorArray,
                outputs_ta: tf.TensorArray,
                outputs_counts_ta: tf.TensorArray,
                return_value: tf.Tensor):

            # Rebuild `stack` tuple
            # TODO: Find way to avoid this by putting tuples into `tf.while_loop` arguments
            stack = stack_1, stack_2

            # Peek into stack by popping but not updating `stack`
            state, hidden_vector, popped_stack = state_stack.pop(stack)

            # Get the summary for all hidden vectors excluding this one
            hidden_vector_summary = state_stack.get_hidden_vector_summary(popped_stack)

            # Get the number of outputs for padding
            # TODO: Doing this twice, once here, and once when calling create_guess_layers. Fix this
            num_outputs = tf.case(
                pred_fn_pairs=[(
                    tf.equal(state, i),
                    lambda i=i: tf.constant(self.object_type.get_all_states()[i].num_outputs))
                    for i in range(len(self.object_type.get_all_states()))],
                default=lambda: tf.constant(0))

            # Call `create_guess_layers(...)` depending on what state we're in
            next_hidden_vector, current_choice = tf.case(
                pred_fn_pairs=[(
                    tf.equal(state, i),
                    lambda i=i: create_guess_layers(hidden_vector_summary, return_value, hidden_vector, i))
                    for i in range(len(self.object_type.get_all_states()))],
                default=lambda: (hidden_vector, tf.constant(0, dtype=tf.float32) / tf.constant(0, tf.float32)))

            # Reshape the hidden vector so we know what size it is
            next_hidden_vector = tf.reshape(next_hidden_vector, [self.hidden_vector_size])

            if self.training:
                # If we're training, the choice we send to the update_state_stack_fn should be determined by the truth
                stack_update_choice = truth_outputs_padded[step][:truth_outputs_counts[step]]
            else:
                # Otherwise, the choice should be what we outputted
                stack_update_choice = current_choice

            # Update the state stack
            stack, return_value = self.__update_state_stack(stack, next_hidden_vector, stack_update_choice)

            return \
                step + 1, \
                (*stack), \
                states_ta.write(step, state), \
                outputs_ta.write(step, current_choice), \
                outputs_counts_ta.write(step, num_outputs), \
                return_value

        def cond(step, stack_1, stack_2, *_):
            # Rebuild `stack` tuple
            stack = stack_1, stack_2

            if self.training:
                return step < step_count
            else:
                return tf.logical_and(step < self.max_steps, tf.not_equal(state_stack.is_empty(stack), True))

        # Create the initial stack with initial hidden vector and state
        initial_stack = state_stack.create(max_size=self.max_steps, hidden_vector_size=self.hidden_vector_size)
        initial_stack = state_stack.push(initial_stack, self.object_type.get_initial_state().id, initial_hidden_vector)

        # Create the `tf.TensorArray` to hold all outputs
        initial_states_ta = tf.TensorArray(
            dtype=tf.float32, size=step_count if self.training else self.max_steps, name="initial_states_ta")
        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32, size=step_count if self.training else self.max_steps, name="initial_outputs_ta")
        initial_outputs_counts_ta = tf.TensorArray(
            dtype=tf.int32, size=step_count if self.training else self.max_steps, name="initial_outputs_counts_ta")

        initial_return_value = tf.constant(np.nan, dtype=tf.float32, shape=[self.hidden_vector_size])

        final_step, *_, final_states_ta, final_outputs_ta, final_outputs_counts_ta, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                0,
                *initial_stack,
                initial_states_ta,
                initial_outputs_ta,
                initial_outputs_counts_ta,
                initial_return_value],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, self.hidden_vector_size + 1]),
                tf.TensorShape([]),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape([self.hidden_vector_size])],
            name="step_while_loop")

        # If in test mode, we don't know the amount of steps we will execute. So we need to resize the `tf.TensorArray`
        # to match the actual output
        # TODO: Check if we can overcome this by using padding
        if not self.training:
            final_states_ta = tf_utils.resize_tensor_array(final_states_ta, final_step)
            final_outputs_ta = tf_utils.resize_tensor_array(final_outputs_ta, final_step)
            final_outputs_counts_ta = tf_utils.resize_tensor_array(final_outputs_counts_ta, final_step)

        return final_states_ta, final_outputs_ta, final_outputs_counts_ta, final_step

    def __update_state_stack(
            self,
            stack: state_stack.StateStack,
            hidden_vector: tf.Tensor,
            output: tf.Tensor) -> (state_stack.StateStack, tf.Tensor):

        state, popped_hidden_vector, stack = state_stack.pop(stack)

        all_state_transitions = self.object_type.get_all_state_transitions()
        pred_fn_pairs = [_state_transition.get_pred_fn_pair(state, output, stack, hidden_vector)
                         for _state_transition in all_state_transitions]

        tensor, element_count, should_return_value = tf.case(
            pred_fn_pairs=pred_fn_pairs,
            default=lambda: (*stack, state_transition.get_should_return_value(None)),
            exclusive=True)

        # TODO: Is this the best place to do this? Should this function be more abstract?
        tensor = tf.reshape(tensor, tf.shape(stack[0]))

        return_value = tf.cond(
            should_return_value,
            # The return value is the popped hidden vector
            lambda: popped_hidden_vector,
            # Else return a `tf.Tensor` of NaNs
            # TODO: Check for other ways of representing the absence of a `tf.Tensor`
            lambda: tf.constant(np.nan, dtype=tf.float32, shape=[state_stack.get_hidden_vector_size(stack)]))

        return (tensor, element_count), return_value

    @staticmethod
    def __get_cost(truth_outputs_padded, generated_outputs_padded):
        with tf.variable_scope("cost"):
            tf.assert_equal(tf.shape(truth_outputs_padded), tf.shape(generated_outputs_padded))

            return tf.sqrt(
                tf.reduce_mean(tf.square(tf.abs(truth_outputs_padded - generated_outputs_padded))), name="cost")

    @staticmethod
    def __pad_ta_elements(ta: tf.TensorArray, size: int) -> tf.TensorArray:
        with tf.variable_scope("pad_ta_elements"):
            def body(step, padded_ta):
                current = ta.read(step)
                padding_shape = tf.concat([[size - tf.shape(current)[0]], tf.shape(current)[1:]], axis=0)
                current_padded = tf.concat([current, tf.zeros(padding_shape, dtype=ta.dtype)], axis=0)

                padded_ta = padded_ta.write(step, current_padded)

                return step + 1, padded_ta

            _, ta = tf.while_loop(
                cond=lambda step, _: step < ta.size(),
                body=body,
                loop_vars=[0, tf.TensorArray(dtype=ta.dtype, size=ta.size())])

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
