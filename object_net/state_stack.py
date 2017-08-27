from typing import Union, Tuple
import tensorflow as tf


StateStack = Tuple[tf.Tensor, tf.Tensor]
"""
The stack object where the `tf.TensorArray` contains the stack elements, and `tf.Tensor` is a single int32 that
represents the current length of the stack
"""


def create(max_size: int, hidden_vector_size: int) -> StateStack:
    """
    Create a new `StateStack`
    :param max_size: the maximum amount of state items in the stack
    :param hidden_vector_size: the size of hidden vectors in the stack
    :return: the new `StateStack`
    """
    return \
        tf.zeros([max_size, hidden_vector_size + 1], dtype=tf.float32, name="state_stack"), \
        tf.constant(0, dtype=tf.int32)


def push(state_stack: StateStack, pushed_state: Union[tf.Tensor, int], pushed_hidden_vector: tf.Tensor) -> StateStack:
    """
    Push an item on the stack
    :param state_stack: the stack to put the items on to
    :param pushed_state: the state number to push on
    :param pushed_hidden_vector: the hidden vector to push on
    :return: the new state stack
    """
    tensor, element_count = state_stack

    with tf.control_dependencies([tf.assert_less_equal(element_count, get_max_size(state_stack))]):
        new_element_count = tf.add(element_count, 1)
        new_tensor = tf.concat(
            [
                tensor[0:element_count],
                [tf.concat([[pushed_state], pushed_hidden_vector], axis=0, name="make_state")],
                tensor[new_element_count:]],
            axis=0,
            name="make_state_stack")

        return new_tensor, new_element_count


def pop(state_stack: StateStack) -> (tf.Tensor, tf.Tensor, StateStack):
    """
    Get the next item off the stack
    :param state_stack: the stack to pop
    :return: the popped state, the popped hidden vector, and the new `StateStack`
    """
    tensor, element_count = state_stack

    with tf.control_dependencies([tf.assert_greater(element_count, 0)]):
        new_element_count = tf.subtract(element_count, 1)
        popped_element = tensor[new_element_count]
        popped_state, popped_hidden_vector = tf.split(popped_element, [1, get_hidden_vector_size(state_stack)], 0)

        popped_state = tf.reshape(popped_state, [])
        popped_hidden_vector = tf.reshape(popped_hidden_vector, [get_hidden_vector_size(state_stack)])

        return popped_state, popped_hidden_vector, (tensor, new_element_count)


def is_empty(state_stack: StateStack) -> tf.Tensor:
    """
    Check if the stack is empty
    :param state_stack: the stack to check
    :return: True if there are no items in the stack, false otherwise
    """
    _, element_count = state_stack

    return tf.equal(element_count, 0)


def get_hidden_vector_summary(state_stack: StateStack) -> tf.Tensor:
    tensor, element_count = state_stack

    occupied_stack = tf.slice(tensor, [0, 1], [element_count, -1])

    return tf.cond(
        tf.not_equal(element_count, 0),
        lambda: tf.reduce_mean(occupied_stack, axis=0),
        lambda: tf.zeros(shape=[get_hidden_vector_size(state_stack)], dtype=tensor.dtype))


def get_max_size(state_stack: StateStack):
    """
    Get the maximum amount of elements the stack can take
    :param state_stack: the stackto check
    :return: the max size as a `tf.Tensor`
    """
    tensor, _ = state_stack

    return tf.shape(tensor)[0]


def get_hidden_vector_size(state_stack: StateStack) -> int:
    """
    Get the size of hidden vectors that can be placed into the stack
    :param state_stack: the stack to check
    :return: the hidden vector size as a `tf.Tensor`
    """
    tensor, _ = state_stack

    return tensor.get_shape().as_list()[1] - 1
