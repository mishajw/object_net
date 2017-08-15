import numpy as np
import tensorflow as tf


class PaddedData:
    @classmethod
    def from_unpadded(cls, unpadded: [[(int, [float])]]):
        return cls(*pad(unpadded))

    def __init__(
            self,
            step_counts: np.array,
            outputs_counts: np.array,
            states_padded: np.array,
            outputs_padded: np.array):

        # Assert that all arrays are the same length
        assert(all([len(step_counts) == len(item) for item in [outputs_counts, states_padded, outputs_padded]]))

        self.step_counts = step_counts
        self.outputs_counts = outputs_counts
        self.states_padded = states_padded
        self.outputs_padded = outputs_padded

    def __getitem__(self, item: int):
        return self.step_counts[item], self.outputs_counts[item], self.states_padded[item], self.outputs_padded[item]

    def __len__(self):
        return self.step_counts.shape[0]


class PlaceholderPaddedData:
    def __init__(self):
        self.step_counts = tf.placeholder(dtype=tf.int32, shape=[None], name="step_counts")
        self.outputs_counts = tf.placeholder(dtype=tf.int32, shape=[None, None], name="outputs_counts")
        self.states_padded = tf.placeholder(dtype=tf.int32, shape=[None, None], name="states_padded")
        self.outputs_padded = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="outputs_padded")

    def get_feed_dict(self, padded_data_item):
        step_counts, outputs_counts, states_padded, outputs_padded = padded_data_item

        return {
            self.step_counts: step_counts,
            self.outputs_counts: outputs_counts,
            self.states_padded: states_padded,
            self.outputs_padded: outputs_padded
        }

    @property
    def batch_size(self):
        return tf.shape(self.step_counts)[0]


def pad(data: [[(int, [float])]]) -> (np.array, np.array, np.array, np.array):
    """
    Pad the data so that it has uniform dimensions
    :param data: a batch of the data to pad
    :return: a tuple of four elements:
        1) A list showing the amount of steps in each batch element
        2) A list of lists showing the size of each output for each batch element
        3) A list of padded lists of states
        4) A list of padded lists of padded lists of outputs
    """

    # Separate out the states and the outputs
    batch_states = []
    batch_outputs = []
    for item in data:
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


def unpad(padded_data: PaddedData) -> [[(int, [float])]]:
    def unpad_single(step_count, outputs_count, states_padded, outputs_padded):
        for i in range(step_count):
            num_outputs = outputs_count[i]

            state = states_padded[i]
            outputs = outputs_padded[i][:num_outputs].tolist()

            yield state, outputs

    for data in padded_data:
        yield unpad_single(*data)
