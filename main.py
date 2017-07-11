import configargparse
import numpy as np
import object_net_writer
import prime_factors
import random
import tensorflow as tf
import tf_utils


def main():
    # Handle program arguments
    parser = configargparse.ArgParser()
    parser.add_argument(
        "--config", is_config_file=True, default="./object_net.ini", help="Path of ini configuration file")
    parser.add_argument("--hidden_vector_length", type=int, default=64)
    parser.add_argument("--fully_connected_sizes", type=str, default="256,256")
    tf_utils.generic_runner.add_arguments(parser)
    tf_utils.data_holder.add_arguments(parser)
    prime_factors.add_arguments(parser)
    args = parser.parse_args()

    # Generate data
    print("Generating data...")
    tree_arrays = prime_factors.get_tree_arrays(args)
    random.shuffle(tree_arrays)
    step_counts, outputs_counts, states_padded, outputs_padded = pad_data(tree_arrays)
    data_holder = tf_utils.data_holder.DataHolder(
        args,
        get_data_fn=lambda i: (step_counts[i], outputs_counts[i], states_padded[i], outputs_padded[i]),
        data_length=len(step_counts))
    print("Done")

    # Define graph
    truth_step_counts = tf.placeholder(dtype=tf.int32, shape=[None], name="truth_step_counts")
    truth_outputs_counts = tf.placeholder(dtype=tf.int32, shape=[None, None], name="truth_outputs_counts")
    truth_states_padded = tf.placeholder(dtype=tf.int32, shape=[None, None], name="truth_states_padded")
    truth_outputs_padded = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="truth_outputs_padded")

    with tf.variable_scope("truth_initial_hidden_vector_input"):
        truth_initial_hidden_vector_input = tf.reshape(tf.slice(truth_outputs_padded, [0, 0, 0], [-1, 1, 1]), [-1, 1])

    object_net = object_net_writer.ObjectNetWriter(
        truth_step_counts,
        truth_outputs_counts,
        truth_states_padded,
        truth_outputs_padded,
        truth_initial_hidden_vector_input,
        hidden_vector_size=args.hidden_vector_length,
        fully_connected_sizes=tf_utils.int_array_from_str(args.fully_connected_sizes),
        state_outputs=[1, 3, 1, 1],
        get_next_state_fn=prime_factors.get_next_state)

    tf.summary.scalar("object_net/cost", object_net.cost)
    optimizer = tf.train.AdamOptimizer().minimize(object_net.cost)

    # Run training
    def train_step(session, step, training_input, _, summary_writer, all_summaries):
        _step_counts, _outputs_counts, _states_padded, _outputs_padded = training_input

        _, all_summaries = session.run(
            [optimizer, all_summaries],
            {
                truth_step_counts: _step_counts,
                truth_outputs_counts: _outputs_counts,
                truth_states_padded: _states_padded,
                truth_outputs_padded: _outputs_padded
            })

        summary_writer.add_summary(all_summaries, step)

    def test_step(session, step, testing_input, _, summary_writer, all_summaries):
        _step_counts, _outputs_counts, _states_padded, _outputs_padded = testing_input

        cost_result, all_summaries = session.run(
            [object_net.cost, all_summaries],
            {
                truth_step_counts: _step_counts,
                truth_outputs_counts: _outputs_counts,
                truth_states_padded: _states_padded,
                truth_outputs_padded: _outputs_padded
            })

        summary_writer.add_summary(all_summaries, step)

        print("Test cost at step %d: %f" % (step, cost_result))

    tf_utils.generic_runner.run_with_test_train_steps(
        args,
        "object_net",
        get_batch_fn=lambda size: (data_holder.get_batch(size), None),
        testing_data=(data_holder.get_test_data(), None),
        test_step_fn=test_step,
        train_step_fn=train_step)


def pad_data(batch_data: [[(int, [float])]]) -> (np.array, np.array, np.array, np.array):
    """
    Pad the data so that it has uniform dimensions
    :param batch_data: a batch of the data to pad
    :return: a tuple of four elements:
        1) A list showing the amount of steps in each batch element
        2) A list of lists showing the size of each output for each batch element
        3) A list of padded lists of states
        4) A list of padded lists of padded lists of outputs
    """

    # Separate out the states and the outputs
    batch_states = []
    batch_outputs = []
    for item in batch_data:
        batch_states.append([state for state, _ in item])
        batch_outputs.append([outputs for _, outputs in item])

    def pad_1d(l: [], size: int):
        if len(l) == size:
            return l

        padding = [0] * (size - len(l))
        return l + padding

    def pad_2d(ls: [[]], size_1: int, size_2: int):
        inner_padded = [pad_1d(l, size_2) for l in ls]

        if len(inner_padded) == size_1:
            return inner_padded

        return inner_padded + [[0] * size_2] * (size_1 - len(inner_padded))

    step_counts = [len(states) for states in batch_states]
    outputs_counts = [[len(output) for output in outputs] for outputs in batch_outputs]

    max_steps = max(step_counts)
    max_outputs = max([output_count for output_counts in outputs_counts for output_count in output_counts])

    outputs_counts_padded = [pad_1d(outputs_count, max_steps) for outputs_count in outputs_counts]
    batch_states_padded = [pad_1d(states, max_steps) for states in batch_states]
    batch_outputs_padded = [pad_2d(outputs, max_steps, max_outputs) for outputs in batch_outputs]

    return \
        np.array(step_counts), \
        np.array(outputs_counts_padded), \
        np.array(batch_states_padded), \
        np.array(batch_outputs_padded)


if __name__ == "__main__":
    main()
