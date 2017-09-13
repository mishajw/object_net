from . import states
from . import state_stack
import tensorflow as tf
from typing import Callable, Tuple

OtherPredsFn = Callable[[tf.Tensor], tf.Tensor]
TransitionFn = Callable[[], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]


def get_should_return_value(_state):
    return tf.constant(_state is not None)


def create_pred(
        current_state: tf.Tensor,
        current_output: tf.Tensor,
        initial_state: states.State,
        other_preds_fn: OtherPredsFn) -> tf.Tensor:
    state_pred = tf.equal(current_state, initial_state.id)

    other_preds = None if other_preds_fn is None else other_preds_fn(current_output)

    if other_preds is None:
        return state_pred
    elif other_preds.get_shape().ndims == 0:
        return tf.logical_and(state_pred, other_preds)
    else:
        return tf.logical_and(state_pred, tf.reduce_all(other_preds))


class StateTransition:
    def __init__(self, other_preds_fn: OtherPredsFn):
        self.other_preds_fn = other_preds_fn

    def get_pred_fn_pair(
            self,
            current_state: tf.Tensor,
            current_output: tf.Tensor,
            stack: state_stack.StateStack,
            hidden_vector: tf.Tensor) -> (tf.Tensor, TransitionFn):
        raise NotImplementedError()


class InnerStateTransition(StateTransition):
    def __init__(self, initial_state: states.State, next_state: states.State = None, other_preds_fn: OtherPredsFn=None):
        super().__init__(other_preds_fn)
        self.initial_state = initial_state
        self.next_state = next_state

    def get_pred_fn_pair(
            self,
            current_state: tf.Tensor,
            current_output: tf.Tensor,
            stack: state_stack.StateStack,
            hidden_vector: tf.Tensor) -> (tf.Tensor, TransitionFn):
        new_stack = stack

        if self.next_state is not None:
            new_stack = state_stack.push(new_stack, self.next_state.id, hidden_vector)

        return \
            create_pred(current_state, current_output, self.initial_state, self.other_preds_fn), \
            lambda: (*new_stack, get_should_return_value(self.next_state))


class ChildStateTransition(StateTransition):
    def __init__(
            self,
            initial_state: states.State,
            new_child_state: states.State,
            next_inner_state: states.State = None,
            other_preds_fn: OtherPredsFn=None):
        super().__init__(other_preds_fn)
        self.initial_state = initial_state
        self.new_child_state = new_child_state
        self.next_inner_state = next_inner_state

    def get_pred_fn_pair(
            self,
            current_state: tf.Tensor,
            current_output: tf.Tensor,
            stack: state_stack.StateStack,
            hidden_vector: tf.Tensor) -> (tf.Tensor, TransitionFn):
        def fn():
            new_stack = stack

            if self.next_inner_state is not None:
                new_stack = state_stack.push(new_stack, self.next_inner_state.id, hidden_vector)

            new_stack = state_stack.push(new_stack, self.new_child_state.id, hidden_vector)

            return (*new_stack, get_should_return_value(self.next_inner_state))

        return create_pred(current_state, current_output, self.initial_state, self.other_preds_fn), fn
