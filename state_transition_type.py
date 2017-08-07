class StateTransitionType:
    """
    The possible types of state transitions
    """
    pass


class ContinueObject(StateTransitionType):
    """The next state continues making the current object"""
    def __init__(self, next_state: int):
        self.next_state = next_state


class StartObject(StateTransitionType):
    """The next state makes a new object"""
    def __init__(self, new_object_initial_state: int, current_object_next_state: int):
        self.new_object_initial_state = new_object_initial_state
        self.current_object_next_state = current_object_next_state


class StartObjectThenStop(StateTransitionType):
    """The next state makes a new object, but the current one is finished"""
    def __init__(self, new_object_initial_state: int):
        self.new_object_initial_state = new_object_initial_state


class StopObject(StateTransitionType):
    """The transition is the end of the current object"""
    pass
