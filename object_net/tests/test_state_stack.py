from . import test_utils
from object_net import state_stack
import tensorflow as tf
import unittest


class TestStateStack(unittest.TestCase):
    def test_push_pop(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)
        stack = state_stack.push(stack, test_utils.constant(1), test_utils.constant([1, 1, 1, 1]))

        state, hidden_vector, stack = state_stack.pop(stack)

        with tf.Session() as sess:
            self.assertEqual(sess.run(state), 1)
            self.assertListEqual(sess.run(hidden_vector).tolist(), [1, 1, 1, 1])

    def test_multiple_push_pop(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)
        stack = state_stack.push(stack, test_utils.constant(1), test_utils.constant([1, 1, 1, 1]))
        stack = state_stack.push(stack, test_utils.constant(2), test_utils.constant([2, 2, 2, 2]))
        stack = state_stack.push(stack, test_utils.constant(3), test_utils.constant([3, 3, 3, 3]))

        state3, hidden_vector3, stack = state_stack.pop(stack)
        state2, hidden_vector2, stack = state_stack.pop(stack)
        state1, hidden_vector1, stack = state_stack.pop(stack)

        with tf.Session() as sess:
            self.assertEqual(sess.run(state1), 1)
            self.assertListEqual(sess.run(hidden_vector1).tolist(), [1, 1, 1, 1])
            self.assertEqual(sess.run(state2), 2)
            self.assertListEqual(sess.run(hidden_vector2).tolist(), [2, 2, 2, 2])
            self.assertEqual(sess.run(state3), 3)
            self.assertListEqual(sess.run(hidden_vector3).tolist(), [3, 3, 3, 3])

    def test_peek(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)
        stack = state_stack.push(stack, test_utils.constant(1), test_utils.constant([1, 1, 1, 1]))

        state1, hidden_vector1, _ = state_stack.pop(stack)
        state2, hidden_vector2, _ = state_stack.pop(stack)

        with tf.Session() as sess:
            self.assertEqual(sess.run(state1), sess.run(state2))
            self.assertListEqual(sess.run(hidden_vector1).tolist(), sess.run(hidden_vector2).tolist())

    def test_pop_fail(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)
        state, _, _ = state_stack.pop(stack)

        with tf.Session() as sess:
            self.assertRaises(Exception, lambda: sess.run(state))

    def test_is_empty(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)
        is_empty = state_stack.is_empty(stack)

        with tf.Session() as sess:
            self.assertTrue(sess.run(is_empty))

    def test_is_empty_after_pop(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)
        stack = state_stack.push(stack, test_utils.constant(1), test_utils.constant([1, 1, 1, 1]))
        stack = state_stack.push(stack, test_utils.constant(2), test_utils.constant([2, 2, 2, 2]))
        is_empty_false1 = state_stack.is_empty(stack)
        _, _, stack = state_stack.pop(stack)
        is_empty_false2 = state_stack.is_empty(stack)
        _, _, stack = state_stack.pop(stack)
        is_empty_true = state_stack.is_empty(stack)

        with tf.Session() as sess:
            self.assertTrue(sess.run(is_empty_true))
            self.assertFalse(sess.run(is_empty_false1))
            self.assertFalse(sess.run(is_empty_false2))

    def test_get_hidden_vector_summary(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)
        stack = state_stack.push(stack, test_utils.constant(1), test_utils.constant([1, 1, 1, 1]))
        stack = state_stack.push(stack, test_utils.constant(2), test_utils.constant([2, 2, 2, 2]))
        stack = state_stack.push(stack, test_utils.constant(3), test_utils.constant([3, 3, 3, 3]))

        hidden_vector_summary = state_stack.get_hidden_vector_summary(stack)

        with tf.Session() as sess:
            self.assertEqual(sess.run(tf.shape(hidden_vector_summary)), 4)
            self.assertListEqual(sess.run(hidden_vector_summary).tolist(), [2, 2, 2, 2])

    def test_empty_get_hidden_vector_summary(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)

        hidden_vector_summary = state_stack.get_hidden_vector_summary(stack)

        with tf.Session() as sess:
            self.assertEqual(sess.run(tf.shape(hidden_vector_summary)), 4)
            self.assertListEqual(sess.run(hidden_vector_summary).tolist(), [0, 0, 0, 0])

    def test_get_max_size(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)
        max_size = state_stack.get_max_size(stack)

        with tf.Session() as sess:
            self.assertEqual(sess.run(max_size), 10)

    def test_get_hidden_vector_size(self):
        stack = state_stack.create(max_size=10, hidden_vector_size=4)
        hidden_vector_size = state_stack.get_hidden_vector_size(stack)

        with tf.Session() as sess:
            self.assertEqual(sess.run(hidden_vector_size), 4)


if __name__ == "__main__":
    unittest.main()
