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
        self.max_outputs = tf.shape(truth_padded_data.outputs_padded)[2]

        num_batches = truth_padded_data.batch_size

        with tf.variable_scope("batch_while_loop"):
            final_cost, \
                generated_states_padded_ta, \
                generated_outputs_padded_ta, \
                generated_outputs_counts_padded_ta, \
                generated_step_counts_ta = self.__batch_while_loop(
                    truth_padded_data, initial_hidden_vector_input, num_batches)

        if not self.training:
            self.generated_states_padded = generated_states_padded_ta.stack()
            self.generated_outputs_padded = generated_outputs_padded_ta.stack()
            self.generated_outputs_counts_padded = generated_outputs_counts_padded_ta.stack()
            self.generated_step_counts = generated_step_counts_ta.stack()

        self.cost = final_cost

    def __batch_while_loop(
            self,
            truth_padded_data: padder.PlaceholderPaddedData,
            initial_hidden_vector_input: tf.Tensor,
            num_batches: int) -> (tf.TensorArray, tf.TensorArray):

        def body(
                step,
                total_cost: tf.Tensor,
                batch_states_ta: tf.TensorArray,
                batch_outputs_ta: tf.TensorArray,
                batch_outputs_counts_ta: tf.TensorArray,
                batch_step_counts_ta: tf.TensorArray):

            with tf.variable_scope("initial_hidden_vector"):
                current_initial_hidden_vector_input = tf.gather(
                    initial_hidden_vector_input, step, name="current_initial_hidden_vector_input")
                current_hidden_vector = self.__create_fully_connected_layers(
                    current_initial_hidden_vector_input, [self.hidden_vector_size])

            with tf.variable_scope("step_while_loop"):
                current_step_count = tf.gather(
                    truth_padded_data.step_counts, step, name="current_step_count")
                current_outputs_padded = tf.gather(
                    truth_padded_data.outputs_padded, step, name="current_outputs_padded")
                current_outputs_counts = tf.gather(
                    truth_padded_data.outputs_counts, step, name="current_outputs_counts")

                output_step_count, output_cost, output_states, output_outputs, output_outputs_counts = \
                    self.__step_while_loop(
                        current_step_count,
                        current_outputs_padded,
                        current_outputs_counts,
                        current_hidden_vector)

            if not self.training:
                batch_states_ta = batch_states_ta.write(step, output_states, "write_batch_states")
                batch_outputs_ta = batch_outputs_ta.write(step, output_outputs, "write_batch_outputs")
                batch_outputs_counts_ta = batch_outputs_counts_ta.write(
                    step, current_outputs_counts, "write_batch_outputs_counts")
                batch_step_counts_ta = batch_step_counts_ta.write(step, current_step_count, "write_step_counts")

            return \
                step + 1, \
                tf.add(total_cost, output_cost), \
                batch_states_ta, \
                batch_outputs_ta, \
                batch_outputs_counts_ta, \
                batch_step_counts_ta

        def cond(step, *_):
            return step < num_batches

        *_, \
            final_total_cost, \
            generated_states_padded_ta, \
            generated_outputs_padded_ta, \
            generated_outputs_counts_padded_ta, \
            generated_step_counts_ta = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0, dtype=tf.float32),
                    tf.TensorArray(dtype=tf.float32, size=num_batches, name="batch_states_ta"),
                    tf.TensorArray(dtype=tf.float32, size=num_batches, name="batch_outputs_ta"),
                    tf.TensorArray(dtype=tf.int32, size=num_batches, name="batch_outputs_counts_ta"),
                    tf.TensorArray(dtype=tf.int32, size=num_batches, name="batch_step_counts_ta")])

        # Average the cost across all batches
        final_cost = tf.divide(final_total_cost, tf.cast(num_batches, dtype=tf.float32))

        return \
            final_cost, \
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
                total_cost: tf.TensorArray,
                states_ta: tf.TensorArray,
                outputs_ta: tf.TensorArray,
                outputs_counts_ta: tf.TensorArray,
                return_value: tf.Tensor):

            # stack_1 = tf.Print(stack_1, [step, tf.slice(stack_1, [0, 1], [-1, 2])], "stack: ", summarize=100)

            # Rebuild `stack` tuple
            # TODO: Find way to avoid this by putting tuples into `tf.while_loop` arguments
            stack = stack_1, stack_2

            # Get the state and hidden vector we have to deal with for this iteration
            state, hidden_vector, stack = state_stack.pop(stack)

            # Get the summary for all hidden vectors excluding this one
            hidden_vector_summary = state_stack.get_hidden_vector_summary(stack)

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

            # Zero pad the current choice
            current_choice = tf.concat(
                [current_choice, tf.zeros([self.max_outputs - tf.shape(current_choice)[0]])],
                axis=0,
                name="current_choice_zero_padded")

            # Reshape the hidden vector so we know what size it is
            next_hidden_vector = tf.reshape(
                next_hidden_vector, [self.hidden_vector_size], name="next_hidden_vector_reshaped")

            truth_current_choice = tf.gather(truth_outputs_padded, step, name="truth_current_choice")

            with tf.variable_scope("current_cost"):
                current_cost = tf.reduce_sum(tf.squared_difference(truth_current_choice, current_choice))
                # Divide by the number of outputs at this step
                # TODO: Check if tf.cast is needed
                current_cost = tf.divide(current_cost, tf.cast(num_outputs, dtype=tf.float32))

            if self.training:
                # If we're training, the choice we send to the update_state_stack_fn should be determined by the truth
                stack_update_choice = truth_current_choice
            else:
                # Otherwise, the choice should be what we outputted
                stack_update_choice = current_choice

            # stack = (tf.Print(stack[0], [step, stack[0][:stack_2]], "estack: ", summarize=100), stack[1])

            # Update the state stack
            stack, return_value = self.__update_state_stack(state, stack, next_hidden_vector, stack_update_choice)

            # We only need the output if we're not training
            if not self.training:
                states_ta = states_ta.write(step, state, "write_state")
                outputs_ta = outputs_ta.write(step, current_choice, "write_outputs")
                outputs_counts_ta = outputs_counts_ta.write(step, num_outputs, "write_outputs_count")

            return \
                step + 1, \
                (*stack), \
                tf.add(total_cost, current_cost), \
                states_ta, \
                outputs_ta, \
                outputs_counts_ta, \
                return_value

        def cond(step, stack_1, stack_2, *_):
            # Rebuild `stack` tuple
            stack = stack_1, stack_2

            if self.training:
                return step < step_count
            else:
                return tf.logical_and(step < self.max_steps, tf.not_equal(state_stack.check_size(stack, 1), True))

        # Create the tensor that keeps track of the total cost
        initial_total_cost = tf.constant(0, dtype=tf.float32, name="initial_total_cost")

        # Create the initial stack with initial hidden vector and state
        initial_stack = state_stack.create(max_size=self.max_steps, hidden_vector_size=self.hidden_vector_size)
        initial_stack = state_stack.push(initial_stack, -1, initial_hidden_vector)
        initial_stack = state_stack.push(initial_stack, self.object_type.get_initial_state().id, initial_hidden_vector)

        # If we're training we know the output sizes and don't need larger than necessary `TensorArray`s
        output_sizes = step_count if self.training else self.max_steps

        # Create the `tf.TensorArray` to hold all outputs
        initial_states_ta = tf.TensorArray(
            dtype=tf.float32, size=output_sizes, name="initial_states_ta")
        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32, size=output_sizes, name="initial_outputs_ta")
        initial_outputs_counts_ta = tf.TensorArray(
            dtype=tf.int32, size=output_sizes, name="initial_outputs_counts_ta")

        initial_return_value = tf.constant(np.nan, dtype=tf.float32, shape=[self.hidden_vector_size])

        final_step, *_, final_total_cost, final_states_ta, final_outputs_ta, final_outputs_counts_ta, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                0,
                *initial_stack,
                initial_total_cost,
                initial_states_ta,
                initial_outputs_ta,
                initial_outputs_counts_ta,
                initial_return_value],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, self.hidden_vector_size + 1]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape([self.hidden_vector_size])],
            name="step_while_loop")

        # Average the cost across all steps
        final_cost = tf.divide(final_total_cost, tf.cast(final_step, dtype=tf.float32))

        # If in test mode, we don't know the amount of steps we will execute. So we need to resize the `tf.TensorArray`
        # to match the actual output
        # TODO: Check if we can overcome this by using padding
        if not self.training:
            final_states_ta = tf_utils.resize_tensor_array(final_states_ta, final_step)
            final_outputs_ta = tf_utils.resize_tensor_array(final_outputs_ta, final_step)
            final_outputs_counts_ta = tf_utils.resize_tensor_array(final_outputs_counts_ta, final_step)

            final_states_ta = self.__stack_and_pad(final_states_ta, self.max_steps)
            final_outputs_ta = self.__stack_and_pad(final_outputs_ta, self.max_steps)
            final_outputs_counts_ta = self.__stack_and_pad(final_outputs_counts_ta, self.max_steps)

        return final_step, final_cost, final_states_ta, final_outputs_ta, final_outputs_counts_ta

    def __update_state_stack(
            self,
            state: int,
            stack: state_stack.StateStack,
            hidden_vector: tf.Tensor,
            output: tf.Tensor) -> (state_stack.StateStack, tf.Tensor):

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
            lambda: hidden_vector,
            # Else return a `tf.Tensor` of NaNs
            # TODO: Check for other ways of representing the absence of a `tf.Tensor`
            lambda: tf.constant(np.nan, dtype=tf.float32, shape=[state_stack.get_hidden_vector_size(stack)]))

        return (tensor, element_count), return_value

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

    @staticmethod
    def __stack_and_pad(tensor_array: tf.TensorArray, length: int):
        stacked = tensor_array.stack()
        stacked_shape = tf.shape(stacked)
        padding_size = length - stacked_shape[0]
        stacked_shape_tail = stacked_shape[1:]
        padding = tf.zeros(
            tf.concat([[padding_size], stacked_shape_tail], axis=0),
            dtype=tensor_array.dtype,
            name="tensor_array_padding")

        return tf.concat([stacked, padding], axis=0)

