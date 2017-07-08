import argparse
import numpy as np
import object_net_writer
import prime_factors
import random
import tensorflow as tf
import tf_utils


def main():
    # Handle program arguments
    parser = argparse.ArgumentParser()
    tf_utils.generic_runner.add_arguments(parser)
    tf_utils.data_holder.add_arguments(parser)
    args = parser.parse_args()
    batch_size = args.batch_size

    # Generate data
    print("Generating data...")
    tree_arrays = prime_factors.get_numpy_arrays(10)
    random.shuffle(tree_arrays)
    padded_data, padded_data_real_length = pad_data(tree_arrays)
    data_holder = tf_utils.data_holder.DataHolder(
        args, get_data_fn=lambda i: (padded_data[i], padded_data_real_length[i]), data_length=len(padded_data))
    print("Done")

    # Define graph
    truth_padded = tf.placeholder(dtype=tf.float32, shape=[None, None, 2], name="truth_padded")
    truth_real_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="truth_real_lengths")
    truth_states_ta, truth_outputs_ta = unpad_data(truth_padded, truth_real_lengths)

    object_net = object_net_writer.ObjectNetWriter(
        truth_states_ta,
        truth_outputs_ta,
        hidden_vector_size=32,
        batch_size=batch_size,
        get_next_state_fn=prime_factors.get_next_state)

    tf.summary.scalar("object_net/cost", object_net.cost)
    optimizer = tf.train.AdamOptimizer().minimize(object_net.cost)

    # Run training
    def train_step(session, step, training_input, _, summary_writer, all_summaries):
        input_padded, input_real_lengths = training_input

        _, all_summaries = session.run(
            [optimizer, all_summaries],
            {
                truth_padded: input_padded,
                truth_real_lengths: input_real_lengths
            })

        summary_writer.add_summary(all_summaries, step)

    def test_step(session, step, testing_input, _, summary_writer, all_summaries):
        input_padded, input_real_lengths = testing_input

        cost_result, all_summaries = session.run(
            [object_net.cost, all_summaries],
            {
                truth_padded: input_padded,
                truth_real_lengths: input_real_lengths
            })

        summary_writer.add_summary(all_summaries, step)

        print("Test cost at step %d: %f" % (step, cost_result))

    tf_utils.generic_runner.run_with_test_train_steps(
        args,
        "object_net",
        get_batch_fn=lambda size: (data_holder.get_batch(size), None),
        testing_data=None,  # (data_holder.get_test_data(), None),
        test_step_fn=test_step,
        train_step_fn=train_step)


def pad_data(data: [[[int]]]) -> (np.array, np.array):
    """
    Pad the data in the form [batch, step, state or output]
    :param data: the data to pad
    :return: a tuple where first is the padded data, and the second is the real size of prepadded data
    """

    def pad(d: [[int]], size: int):
        if len(d) == size:
            return d

        padding = [[0, 0]] * (max_length - len(d))
        return np.concatenate([d, padding])

    lengths = [len(d) for d in data]
    max_length = max(lengths)
    padded = [pad(d, max_length) for d in data]

    return np.array(padded), np.array(lengths)


def unpad_data(data: tf.Tensor, data_real_lengths: tf.Tensor) -> (tf.TensorArray, tf.TensorArray):
    """
    Remove the padding from a `tf.Tensor`
    :param data: the padded data
    :param data_real_lengths: the original lengths of the data
    :return: a tuple of the unpadded states and outputs
    """
    data_length = tf.shape(data)[0]
    data_states_ta = tf.TensorArray(dtype=tf.float32, size=data_length)
    data_outputs_ta = tf.TensorArray(dtype=tf.float32, size=data_length)

    def body(
            step,
            _data: tf.Tensor,
            _data_real_lengths: tf.Tensor,
            _data_states_ta: tf.TensorArray,
            _data_outputs_ta: tf.TensorArray):

        current_state = tf.reshape(tf.slice(_data[step], [0, 0], [_data_real_lengths[step], 1]), [-1])
        current_outputs = tf.reshape(tf.slice(_data[step], [0, 1], [_data_real_lengths[step], 1]), [-1])

        _data_states_ta = _data_states_ta.write(step, current_state)
        _data_outputs_ta = _data_outputs_ta.write(step, current_outputs)
        return step + 1, _data, _data_real_lengths, _data_states_ta, _data_outputs_ta

    *_, data_states_ta, data_outputs_ta = tf.while_loop(
        lambda step, *_: step < data_length,
        body=body,
        loop_vars=[0, data, data_real_lengths, data_states_ta, data_outputs_ta])

    return data_states_ta, data_outputs_ta


if __name__ == "__main__":
    main()
