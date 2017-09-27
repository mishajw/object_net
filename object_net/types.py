from . import states, state_transition
from typing import Dict, List, Any, Tuple, Iterator, Callable, Optional
from typing import Type as TypingType
import itertools
import json
import queue
import tensorflow as tf


def assert_string_exists(json_object, name: str):
    assert name in json_object
    assert isinstance(json_object[name], str)


def assert_string_list_exists(json_object: dict, name: str):
    assert name in json_object
    assert isinstance(json_object[name], list)
    assert all(isinstance(option, str) for option in json_object[name])


class Type:
    def __init__(self, name: str, _states: List[states.State]):
        self.name = name
        self.states = _states

    @classmethod
    def from_json(cls, json_object):
        raise NotImplementedError()

    def validate(self, value):
        raise NotImplementedError()

    def get_child_keys(self) -> iter:
        """
        :return: an iterator that gives keys to be used for accessing children types
        """
        return []

    def get_child_type(self, key) -> "Type":
        raise ValueError("No children")

    def set_child_type(self, key, value):
        raise ValueError("No children")

    def get_state_output_pairs(self, value: Any) -> List[Tuple[int, List[float]]]:
        raise NotImplementedError()

    def resolve_references(self, type_dict):
        for key in self.get_child_keys():
            child_type = self.get_child_type(key)
            if isinstance(child_type, ReferenceType):
                if child_type.name in type_dict.keys():
                    self.set_child_type(key, type_dict[child_type.name])
                else:
                    raise UnknownReferenceError(child_type.name)

    def get_state_by_name(self, state_name: str):
        for state in self.states:
            if state.name == state_name:
                return state

        raise ValueError("Can't find state in type %s with name %s" % (self, state_name))

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        raise NotImplementedError()

    def get_initial_state(self):
        if len(self.states) > 0:
            return self.states[0]

        raise ValueError("No states available to return initial state")

    def get_all_states(self) -> List[states.State]:
        all_types = self.get_all_types()
        all_states = [_type.states for _type in all_types]

        return list(itertools.chain(*all_states))

    def get_all_state_transitions(self) -> List[state_transition.StateTransition]:
        all_types = self.get_all_types()
        all_state_transitions = [_type.get_state_transitions() for _type in all_types]

        return list(itertools.chain(*all_state_transitions))

    # TODO: Explore caching this result
    def get_all_types(self) -> List["Type"]:
        explored_types = [self]
        explore_queue = queue.Queue()
        explore_queue.put(self)

        while not explore_queue.empty():
            current_type = explore_queue.get()
            explored_types.append(current_type)

            for key in current_type.get_child_keys():
                child = current_type.get_child_type(key)
                if child not in explored_types:
                    explore_queue.put(child)

        return list(set(explored_types))

    def get_value_from_state_output_pairs(self, state_output_pairs: Iterator[Tuple[int, List[float]]]) -> Any:
        raise NotImplementedError()

    def can_flatten(self):
        return True


TypeDict = Dict[str, Type]


class PrimitiveType(Type):
    def __init__(self, name: str, primitive: TypingType, num_outputs: int, output_type: states.OutputType):
        super().__init__(name, [states.State(name, num_outputs, output_type)])
        self.primitive = primitive

    @classmethod
    def from_json(cls, json_object):
        raise TypeError("Can't create a new primitive type from json")

    def validate(self, value):
        assert isinstance(value, self.primitive)

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        return []

    def get_state_output_pairs(self, value: Any) -> List[Tuple[int, List[float]]]:
        return [(self.get_initial_state().id, [value])]

    def get_value_from_state_output_pairs(self, state_output_pairs: Iterator[Tuple[int, List[float]]]) -> Any:
        state, output = next(state_output_pairs)

        assert state == self.get_initial_state().id
        assert len(output) == 1

        return self.primitive(output[0])


class EnumType(Type):
    def __init__(self, name: str, options: List[str]):
        super().__init__(name, [states.State(name, len(options), states.OutputType.BOOL)])
        self.options = options

    @classmethod
    def from_json(cls, json_object):
        assert_string_exists(json_object, "name")
        assert_string_list_exists(json_object, "options")

        return EnumType(json_object["name"], json_object["options"])

    def validate(self, value):
        assert any([value == option for option in self.options])

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        return []

    def get_state_output_pairs(self, value: Any) -> List[Tuple[int, List[float]]]:
        output = [0] * len(self.options)
        output[self.options.index(value)] = 1

        return [(self.get_initial_state().id, output)]

    def get_value_from_state_output_pairs(self, state_output_pairs: Iterator[Tuple[int, List[float]]]) -> Any:
        state, output = next(state_output_pairs)

        assert state == self.get_initial_state().id
        assert len(output) == len(self.options)

        # TODO: Check if there's an argmax function in stdlib
        max_index = 0
        for i in range(len(output)):
            if output[max_index] < output[i]:
                max_index = i

        return self.options[max_index]


class UnionType(Type):
    def __init__(self, name: str, types: List[Type]):
        super().__init__(name, [states.State(name, len(types), states.OutputType.BOOL)])
        self.types = types

    @classmethod
    def from_json(cls, json_object):
        assert_string_exists(json_object, "name")
        assert_string_list_exists(json_object, "types")

        return UnionType(json_object["name"], [ReferenceType(type_str) for type_str in json_object["types"]])

    def validate(self, value):
        assert [_type.validate(value) for _type in self.types]

    def get_child_keys(self) -> iter:
        return range(len(self.types))

    def get_child_type(self, key) -> "Type":
        return self.types[key]

    def set_child_type(self, key, value):
        self.types[key] = value

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        return [
            state_transition.ChildStateTransition(
                self.get_initial_state(),
                _type.get_initial_state(),
                other_preds_fn=lambda output: tf.equal(tf.argmax(output), i))
            for i, _type in enumerate(self.types)]

    def get_state_output_pairs(self, value: Any) -> List[Tuple[int, List[float]]]:
        raise NotImplementedError()

    def get_value_from_state_output_pairs(self, state_output_pairs: Iterator[Tuple[int, List[float]]]) -> Any:
        raise NotImplementedError()


class OptionalType(Type):
    def __init__(self, _type: Type):
        name = "optional[%s]" % _type.name
        super().__init__(name, [states.State(name, 1, states.OutputType.BOOL)])
        self.type = _type

    @classmethod
    def from_json(cls, json_object):
        assert_string_exists(json_object, "type")

        return OptionalType(ReferenceType(json_object["type"]))

    def validate(self, value):
        if value is not None:
            self.type.validate(value)

    def get_child_keys(self) -> iter:
        return [None]

    def get_child_type(self, key) -> "Type":
        assert key is None
        return self.type

    def set_child_type(self, key, value):
        assert key is None
        self.type = value

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        return [
            state_transition.ChildStateTransition(
                self.get_initial_state(),
                self.type.get_initial_state(),
                other_preds_fn=lambda output: condition_if_exists(output, lambda: tf.greater_equal(output[0], 0.5))),
            state_transition.InnerStateTransition(
                self.get_initial_state(),
                other_preds_fn=lambda output: condition_if_exists(output, lambda: tf.less(output[0], 0.5)))]

    def get_state_output_pairs(self, value: Any) -> List[Tuple[int, List[float]]]:
        value_is_empty = value is None

        optional_state = (self.get_initial_state().id, [0.0 if value_is_empty else 1.0])

        if value_is_empty:
            return [optional_state]
        else:
            return [optional_state] + list(self.type.get_state_output_pairs(value))

    def get_value_from_state_output_pairs(self, state_output_pairs: Iterator[Tuple[int, List[float]]]) -> Any:
        state, output = next(state_output_pairs)

        assert state == self.get_initial_state().id
        assert len(output) == 1

        if output[0] < 0.5:
            return None
        else:
            return self.type.get_value_from_state_output_pairs(state_output_pairs)


class ObjectType(Type):
    def __init__(self, name: str, fields: Dict[str, Type]):
        super().__init__(name, [
            states.State(self.__get_state_name(key, name), 0, states.OutputType.NONE)
            for key in fields])
        self.fields = fields
        self.__first_field_key = next(iter(self.fields))

    @classmethod
    def from_json(cls, json_object: dict):
        assert_string_exists(json_object, "name")

        fields = dict(
            (field_name, ReferenceType(json_object[field_name])) for field_name in json_object if field_name != "name")

        return ObjectType(json_object["name"], fields)

    def validate(self, value):
        assert isinstance(value, dict)

        for field_name in value:
            assert field_name in self.fields.keys()
            self.fields[field_name].validate(value[field_name])

    def get_child_keys(self) -> iter:
        return self.fields.keys()

    def get_child_type(self, key) -> "Type":
        return self.fields[key]

    def set_child_type(self, key, value):
        self.fields[key] = value

    def resolve_references(self, type_dict: TypeDict):
        for key in self.fields.keys():
            _type = self.fields[key]
            if isinstance(_type, ReferenceType):
                if _type.name in type_dict.keys():
                    self.fields[key] = type_dict[_type.name]
                else:
                    raise UnknownReferenceError(_type.name)

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        fields_list = list(self.fields)

        for i in range(len(self.fields.keys()) - 1):
            current_field_name = fields_list[i]
            next_field_name = fields_list[i + 1]

            yield state_transition.ChildStateTransition(
                self.get_state_by_name(self.__get_state_name(current_field_name)),
                self.fields[current_field_name].get_initial_state(),
                self.get_state_by_name(self.__get_state_name(next_field_name)))

        final_field_name = fields_list[-1]
        yield state_transition.ChildStateTransition(
            self.get_state_by_name(self.__get_state_name(final_field_name)),
            self.fields[final_field_name].get_initial_state())

    def __get_state_name(self, field_name, name: str=None):
        if name is None:
            name = self.name

        return "%s.%s" % (name, field_name)

    def get_state_output_pairs(self, value: Any) -> List[Tuple[int, List[float]]]:
        for key in self.fields:
            yield (self.get_state_by_name(self.__get_state_name(key)).id, [])
            yield from self.fields[key].get_state_output_pairs(value[key])

    def get_value_from_state_output_pairs(self, state_output_pairs: Iterator[Tuple[int, List[float]]]) -> Any:
        value = {}
        keys = list(self.fields.keys())

        for i in range(len(keys)):
            state, output = next(state_output_pairs)
            _type = self.fields[keys[i]]
            type_state = self.get_state_by_name(self.__get_state_name(keys[i]))

            assert state == type_state.id
            assert len(output) == 0

            value[keys[i]] = _type.get_value_from_state_output_pairs(state_output_pairs)

        return value

    def can_flatten(self):
        return False


class ReferenceType(Type):
    def __init__(self, name: str):
        super().__init__(name, [])

    @classmethod
    def from_json(cls, json_object):
        raise TypeError("Can't create a new reference type from json")

    def validate(self, value):
        raise TypeError("Can't validate reference type, must resolve all references first")

    def resolve_references(self, type_dict: TypeDict):
        raise TypeError("Can't resolve references on a reference type")

    def get_child_type(self, key) -> "Type":
        raise TypeError("Can't get child type on a reference type")

    def set_child_type(self, key, value):
        raise TypeError("Can't set child type on a reference type")

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        return []

    def get_state_output_pairs(self, value: Any) -> List[Tuple[int, List[float]]]:
        raise TypeError("Can't get state output pairs from reference type")

    def get_value_from_state_output_pairs(self, state_output_pairs: Iterator[Tuple[int, List[float]]]) -> Any:
        raise TypeError("Can't get value from state output pairs on a reference type")


class FlattenType(Type):
    def __init__(self, name: str, _type: Type, return_state: Optional[states.State]):
        assert _type.can_flatten()

        self.state_dict = dict([(state, state.copy()) for state in _type.states])
        for child_state in self.state_dict:
            _state = self.state_dict[child_state]
            _state.name = "%s[%s]" % (child_state.name, _state)

        super().__init__(name, list(self.state_dict.values()))
        self.type = _type
        self.return_state = return_state

    def get_state_output_pairs(self, value: Any) -> List[Tuple[int, List[float]]]:
        child_state_output_pairs = self.type.get_state_output_pairs(value)
        state_output_pairs = []

        for state, output in child_state_output_pairs:
            matching_states = list(filter(lambda s: s.id == state, self.state_dict.keys()))

            if len(matching_states) == 0:
                state_output_pairs.append((state, output))
            elif len(matching_states) == 1:
                new_state = self.state_dict[matching_states[0]]
                state_output_pairs.append((new_state.id, output))

        return state_output_pairs

    @classmethod
    def from_json(cls, json_object):
        raise TypeError()

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        child_state_transitions = self.type.get_state_transitions()
        state_transitions = []

        def replace_state(state: states.State):
            if state in self.state_dict.keys():
                return self.state_dict[state]
            else:
                return state

        # Copy the child's state transitions but replace with our states
        for child_state_transition in child_state_transitions:
            if isinstance(child_state_transition, state_transition.InnerStateTransition):
                state_transitions.append(state_transition.InnerStateTransition(
                    replace_state(child_state_transition.initial_state),
                    replace_state(child_state_transition.next_state),
                    child_state_transition.other_preds_fn))
            elif isinstance(child_state_transition, state_transition.ChildStateTransition):
                state_transitions.append(state_transition.ChildStateTransition(
                    replace_state(child_state_transition.initial_state),
                    replace_state(child_state_transition.new_child_state),
                    replace_state(child_state_transition.next_inner_state),
                    child_state_transition.other_preds_fn))

        # Make the last transition go to the return state
        if len(state_transitions) > 1:
            state_transitions[-1].next_state = self.return_state

        return state_transitions

    def get_value_from_state_output_pairs(self, state_output_pairs: Iterator[Tuple[int, List[float]]]) -> Any:
        pass

    def validate(self, value):
        pass


class UnknownReferenceError(RuntimeError):
    def __init__(self, name: str):
        super().__init__("Couldn't find %s" % name)


class Instance:
    def __init__(self, value: Any, _type: Type):
        _type.validate(value)

        self.value = value
        self.type = _type


int_type = PrimitiveType("int", int, num_outputs=1, output_type=states.OutputType.REAL)
float_type = PrimitiveType("float", float, num_outputs=1, output_type=states.OutputType.REAL)
bool_type = PrimitiveType("bool", bool, num_outputs=1, output_type=states.OutputType.REAL)


def resolve_references(types: List[Type]):
    # Get all unique types
    all_types = list(set(itertools.chain(*[_type.get_all_types() for _type in types])))
    # Add in primitives
    all_types.extend([int_type, float_type, bool_type])
    # Remove reference types
    all_types = list(filter(lambda _type: not isinstance(_type, ReferenceType), all_types))
    # Build dictionary of type names to types
    all_types_dict = dict((_type.name, _type) for _type in all_types if not isinstance(_type, ReferenceType))

    # Remove references in all types
    for _type in all_types:
        _type.resolve_references(all_types_dict)


def create_from_json(json_str: str) -> List[Type]:
    json_object = json.loads(json_str)

    return create_from_dict(json_object)


def create_from_dict(json_object: dict) -> List[Type]:
    assert "types" in json_object and isinstance(json_object["types"], list)

    def get_types():
        for type_object in json_object["types"]:
            assert "base" in type_object and isinstance(type_object["base"], str)
            base = type_object["base"]
            type_object.pop("base")

            if base == "enum":
                yield EnumType.from_json(type_object)
            elif base == "union":
                yield UnionType.from_json(type_object)
            elif base == "optional":
                yield OptionalType.from_json(type_object)
            elif base == "object":
                yield ObjectType.from_json(type_object)

    # Get types
    all_types = list(get_types())

    # Remove all reference types
    resolve_references(all_types)

    # Assign IDs to all states
    if len(all_types) > 0:
        states.State.assign_ids(all_types[0].get_all_states())

    return all_types


def condition_if_exists(output: tf.Tensor, condition: Callable[[], tf.Tensor]) -> tf.Tensor:
    """
    Perform the condition `condition` on `output` if output is not empty
    If `output` is empty, return false
    :param output: the potentially empty tensor to check with `condition`
    :param condition: the condition that takes into account `output`
    :return: a bool typed tensor
    """
    return tf.cond(
        pred=tf.reduce_all(tf.equal(tf.shape(output), [0])),
        fn1=lambda: tf.constant(False),
        fn2=condition)
