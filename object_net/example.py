from . import object_net_components
from . import object_net_writer
from . import padder
from . import prime_factors
import configargparse
import math
import random
import tensorflow as tf
import tf_utils


def example():
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
    trees = prime_factors.get_trees(args)
    arrays = [prime_factors.tree_to_array(tree, args) for tree in trees]
    random.shuffle(arrays)
    padded_arrays = padder.PaddedData.from_unpadded(arrays)
    data_holder = tf_utils.data_holder.DataHolder(
        args,
        get_data_fn=lambda i: padded_arrays[i],
        data_length=len(padded_arrays))
    print("Done")

    # Define graph
    truth_padded_data = padder.PlaceholderPaddedData()

    with tf.variable_scope("truth_initial_hidden_vector_input"):
        truth_initial_hidden_vector_input = tf.reshape(
            tf.slice(truth_padded_data.outputs_padded, [0, 0, 0], [-1, 1, 1]),
            [-1])

    def get_object_net_writer(training: bool) -> object_net_writer.ObjectNetWriter:
        return object_net_writer.ObjectNetWriter(
            truth_padded_data,
            truth_initial_hidden_vector_input,
            hidden_vector_size=args.hidden_vector_length,
            fully_connected_sizes=tf_utils.int_array_from_str(args.fully_connected_sizes),
            state_outputs=[1, 3, 1, 1],
            update_state_stack_fn=prime_factors.update_state_stack,
            initial_state=1,
            training=training,
            inner_hidden_vector_creator=object_net_components.LstmInnerHiddenVectorCreator(args.hidden_vector_length),
            child_hidden_vector_combiner=object_net_components.AdditionChildHiddenVectorCombiner())

    object_net = get_object_net_writer(training=True)
    object_net_test = get_object_net_writer(training=False)

    tf.summary.scalar("object_net/cost", object_net.cost)
    optimizer = tf.train.AdamOptimizer().minimize(object_net.cost)

    # Run training
    def train_step(session, step, training_input, _, summary_writer, all_summaries):
        _, all_summaries = session.run(
            [optimizer, all_summaries],
            truth_padded_data.get_feed_dict(training_input))

        summary_writer.add_summary(all_summaries, step)

    def test_step(session, step, testing_input, _, summary_writer, all_summaries):
        cost_result, \
            generated_states_padded, \
            generated_outputs_padded, \
            generated_outputs_counts_padded, \
            generated_step_counts, \
            current_initial_hidden_vector_input, \
            all_summaries = session.run(
                [
                    object_net_test.cost,
                    object_net_test.generated_states_padded,
                    object_net_test.generated_outputs_padded,
                    object_net_test.generated_outputs_counts_padded,
                    object_net_test.generated_step_counts,
                    truth_initial_hidden_vector_input,
                    all_summaries],
                truth_padded_data.get_feed_dict(testing_input))

        copied_testing_input = padder.PaddedData(
            generated_step_counts, generated_outputs_counts_padded, generated_states_padded, generated_outputs_padded)
        unpadded = padder.unpad(copied_testing_input)

        def try_array_to_tree(_array, _args):
            try:
                return prime_factors.array_to_tree(_array, _args)
            except StopIteration:
                return prime_factors.PrimeFactorTree(-1, None, None)

        generated_trees = [try_array_to_tree(array, args) for array in unpadded]

        for tree, number in list(zip(generated_trees, current_initial_hidden_vector_input))[:10]:
            number = math.pow(math.e, number)
            print("%f -> %s" % (number, tree))

        summary_writer.add_summary(all_summaries, step)

        print("Test cost at step %d: %f" % (step, cost_result))

    tf_utils.generic_runner.run_with_test_train_steps(
        args,
        "object_net",
        get_batch_fn=lambda size: (data_holder.get_batch(size), None),
        testing_data=(data_holder.get_test_data(), None),
        test_step_fn=test_step,
        train_step_fn=train_step)

