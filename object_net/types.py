from . import states, state_transition
from typing import Dict, List, Any, Set
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

        if len(self.states) > 0:
            self.initial_state = self.states[0]
        else:
            self.initial_state = None

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

    # TODO: Look into having polymorphic child setters and getters so this doesn't need to be implemented for all types
    def resolve_references(self, type_dict):
        for key in self.get_child_keys():
            child_type = self.get_child_type(key)
            if isinstance(child_type, ReferenceType):
                if child_type.name in type_dict.keys():
                    self.set_child_type(key, type_dict[child_type.name])
                else:
                    raise UnknownReferenceError()

    def get_state_by_name(self, state_name: str):
        for state in self.states:
            if state.name == state_name:
                return state

        raise ValueError("Can't find state in type %s with name %s" % (self, state_name))

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        raise NotImplementedError()


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
                self.initial_state,
                _type.initial_state,
                other_preds_fn=lambda output: tf.equal(tf.argmax(output), i))
            for i, _type in enumerate(self.types)]


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
                self.initial_state,
                self.type.initial_state,
                other_preds_fn=lambda output: tf.greater_equal(output[0], 0.5)),
            state_transition.InnerStateTransition(
                self.initial_state,
                other_preds_fn=lambda output: tf.less(output[0], 0.5))]


class ObjectType(Type):
    def __init__(self, name: str, fields: Dict[str, Type]):
        super().__init__(name, [])
        self.fields = fields

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
                    raise UnknownReferenceError()

    def get_state_transitions(self) -> List[state_transition.StateTransition]:
        fields_list = list(self.fields)

        for i in range(len(self.fields.keys()) - 1):
            current_field_name = fields_list[i]
            next_field_name = fields_list[i + 1]

            yield state_transition.InnerStateTransition(
                self.fields[current_field_name].initial_state,
                self.fields[next_field_name].initial_state)


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


class UnknownReferenceError(RuntimeError):
    pass


class Instance:
    def __init__(self, value: Any, _type: Type):
        _type.validate(value)

        self.value = value
        self.type = _type


int_type = PrimitiveType("int", int, num_outputs=0, output_type=states.OutputType.REAL)
float_type = PrimitiveType("float", float, num_outputs=0, output_type=states.OutputType.REAL)
bool_type = PrimitiveType("bool", bool, num_outputs=0, output_type=states.OutputType.REAL)


def resolve_references(types: List[Type]):
    types.extend([int_type, float_type, bool_type])

    type_dict = dict((_type.name, _type) for _type in types)

    for _type in types:
        _type.resolve_references(type_dict)


def get_all_state_transitions(_type: Type) -> Set[state_transition.StateTransition]:
    all_types = get_all_types(_type)
    all_state_transitions = [_type.get_state_transitions() for _type in all_types]

    return set(itertools.chain(all_state_transitions))


def get_all_types(_type: Type):
    explored_types = [_type]
    explore_queue = queue.Queue()
    explore_queue.put(_type)

    while not explore_queue.empty():
        current_type = explore_queue.get()
        explored_types.append(current_type)

        for key in current_type.get_child_keys():
            child = current_type.get_child_type(key)
            if child not in explored_types:
                explore_queue.put(child)

    return explored_types


def create_from_json(json_str: str) -> List[Type]:
    json_object = json.loads(json_str)

    return create_from_dict(json_object)


def create_from_dict(json_object: dict) -> List[Type]:
    assert "types" in json_object and isinstance(json_object["types"], list)

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
