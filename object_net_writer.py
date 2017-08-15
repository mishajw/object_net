import object_net_components
import padder
import state_stack
import states
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
            hidden_vector_size: int,
            fully_connected_sizes: [int],
            state_outputs: [int],
            update_state_stack_fn: states.UpdateStateStackFn,
            initial_state: int,
            training: bool,
            inner_hidden_vector_creator: object_net_components.InnerHiddenVectorCreator,
            child_hidden_vector_combiner: object_net_components.ChildHiddenVectorCombiner):
        """
        Initialise TensorFlow graph
        :param truth_padded_data: the input data
        :param initial_hidden_vector_input: the inputs for each example in the batch
        :param hidden_vector_size: size of hidden vectors
        :param fully_connected_sizes: the sizes for fully connected layers
        :param state_outputs: the respective sizes of outputs for each state
        :param update_state_stack_fn: function that gives transitions between states
        """
        self.hidden_vector_size = hidden_vector_size
        self.fully_connected_sizes = fully_connected_sizes
        self.state_outputs = [0] + state_outputs
        self.update_state_stack_fn = update_state_stack_fn
        self.initial_state = initial_state
        self.training = training
        self.inner_hidden_vector_creator = inner_hidden_vector_creator
        self.child_hidden_vector_combiner = child_hidden_vector_combiner

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

            current_step_count = truth_padded_data.step_counts[step]
            current_states_padded = truth_padded_data.states_padded[step]
            current_initial_hidden_vector_input = initial_hidden_vector_input[step]

            with tf.variable_scope("initial_hidden_vector"):
                current_hidden_vector = tf.sigmoid(self.__create_fully_connected_layers(
                    current_initial_hidden_vector_input, self.fully_connected_sizes + [self.hidden_vector_size]))

            current_states_ta, current_outputs_ta, current_outputs_counts_ta, current_step_count = \
                self.__step_while_loop(current_step_count, current_states_padded, current_hidden_vector)

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

    def __step_while_loop(self, step_count: int, states_padded: tf.Tensor, initial_hidden_vector: tf.Tensor):
        def create_guess_layers(hidden_vector: tf.Tensor, hidden_vector_summary: tf.Tensor, state_number: int):
            with tf.variable_scope("guess_layers_%d" % state_number):
                num_outputs = self.state_outputs[state_number]
                return self.inner_hidden_vector_creator(hidden_vector_summary, hidden_vector, num_outputs)

        def body(
                step: int,
                stack_1,
                stack_2,
                states_ta: tf.TensorArray,
                outputs_ta: tf.TensorArray,
                outputs_counts_ta: tf.TensorArray):

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
                    lambda i=i: tf.constant(self.state_outputs[i]))
                    for i in range(len(self.state_outputs))],
                default=lambda: tf.constant(0))

            # Call `create_guess_layers(...)` depending on what state we're in
            next_hidden_vector, current_choice = tf.case(
                pred_fn_pairs=[(
                    tf.equal(state, i),
                    lambda i=i: create_guess_layers(hidden_vector, hidden_vector_summary, i))
                    for i in range(len(self.state_outputs))],
                default=lambda: (hidden_vector, tf.constant(0, dtype=tf.float32) / tf.constant(0, tf.float32)))

            # Reshape the hidden vector so we know what size it is
            next_hidden_vector = tf.reshape(next_hidden_vector, [self.hidden_vector_size])

            if not self.training:
                # Push the next state on the stack based on choice output
                stack = self.update_state_stack_fn(stack, next_hidden_vector, current_choice)
            else:
                # Check we're not on the last step, and then push the next state on the stack from training data
                stack = tf.cond(
                    pred=step < step_count - 1,
                    fn1=lambda: state_stack.push(
                        stack, tf.cast(states_padded[step + 1], dtype=tf.float32), next_hidden_vector),
                    fn2=lambda: stack)

            return \
                step + 1, \
                (*stack), \
                states_ta.write(step, state), \
                outputs_ta.write(step, current_choice), \
                outputs_counts_ta.write(step, num_outputs)

        def cond(step, stack_1, stack_2, *_):
            # Rebuild `stack` tuple
            stack = stack_1, stack_2

            if self.training:
                return step < step_count
            else:
                return tf.logical_and(step < self.max_steps, tf.not_equal(state_stack.is_empty(stack), True))

        # Create the initial stack with initial hidden vector and state
        initial_stack = state_stack.create(max_size=self.max_steps, hidden_vector_size=self.hidden_vector_size)
        initial_stack = state_stack.push(initial_stack, self.initial_state, initial_hidden_vector)

        # Create the `tf.TensorArray` to hold all outputs
        initial_states_ta = tf.TensorArray(
            dtype=tf.float32, size=step_count if self.training else self.max_steps, name="initial_states_ta")
        initial_outputs_ta = tf.TensorArray(
            dtype=tf.float32, size=step_count if self.training else self.max_steps, name="initial_outputs_ta")
        initial_outputs_counts_ta = tf.TensorArray(
            dtype=tf.int32, size=step_count if self.training else self.max_steps, name="initial_outputs_counts_ta")

        final_step, *_, final_states_ta, final_outputs_ta, final_outputs_counts_ta = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[0, *initial_stack, initial_states_ta, initial_outputs_ta, initial_outputs_counts_ta],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, self.hidden_vector_size + 1]),
                tf.TensorShape([]),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape(None)],
            name="step_while_loop")

        # If in test mode, we don't know the amount of steps we will execute. So we need to resize the `tf.TensorArray`
        # to match the actual output
        # TODO: Check if we can overcome this by using padding
        if not self.training:
            final_states_ta = tf_utils.resize_tensor_array(final_states_ta, final_step)
            final_outputs_ta = tf_utils.resize_tensor_array(final_outputs_ta, final_step)
            final_outputs_counts_ta = tf_utils.resize_tensor_array(final_outputs_counts_ta, final_step)

        return final_states_ta, final_outputs_ta, final_outputs_counts_ta, final_step

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
