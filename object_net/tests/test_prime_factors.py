from object_net.tests import test_utils
from object_net import state_stack, prime_factors
import tensorflow as tf
import unittest


class TestPrimeFactors(unittest.TestCase):
    state_encoder = prime_factors.state_encoder

    def test_update_state_stack(self):
        stack = state_stack.create(100, 1)
        stack = state_stack.push(
            stack, TestPrimeFactors.state_encoder.encode("value"), test_utils.constant([1]))

        stack = prime_factors.update_state_stack(stack, test_utils.constant([2]), test_utils.constant([0]))
        mod_three_state, mod_three_hidden_vector, _ = state_stack.pop(stack)

        stack = prime_factors.update_state_stack(stack, test_utils.constant([3]), test_utils.constant([0]))
        left_opt_state, left_opt_hidden_vector, _ = state_stack.pop(stack)

        stack = prime_factors.update_state_stack(stack, test_utils.constant([4]), test_utils.constant([1]))
        # This time store the popped stack so we don't have to run through the inner child
        inner_value_state, inner_value_hidden_vector, stack = state_stack.pop(stack)

        stack = prime_factors.update_state_stack(stack, test_utils.constant([5]), test_utils.constant([0]))
        _, final_stack_size = stack

        with tf.Session() as sess:
            self.assertEqual(sess.run(mod_three_state), TestPrimeFactors.state_encoder.encode("mod_three"))
            self.assertEqual(sess.run(mod_three_hidden_vector), 2)

            self.assertEqual(sess.run(left_opt_state), TestPrimeFactors.state_encoder.encode("left_opt"))
            self.assertEqual(sess.run(left_opt_hidden_vector), 3)

            self.assertEqual(sess.run(inner_value_state), TestPrimeFactors.state_encoder.encode("value"))
            self.assertEqual(sess.run(inner_value_hidden_vector), 4)

            self.assertEqual(sess.run(final_stack_size), 0)
