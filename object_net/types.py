import json
from typing import Dict, List, Any
from typing import Type as TypingType


def assert_string_exists(json_object, name: str):
    assert name in json_object
    assert isinstance(json_object[name], str)


def assert_string_list_exists(json_object: dict, name: str):
    assert name in json_object
    assert isinstance(json_object[name], list)
    assert all(isinstance(option, str) for option in json_object[name])


class Type:
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def from_json(cls, json_object):
        raise NotImplementedError()

    def validate(self, value):
        raise NotImplementedError()

    # TODO: Look into having polymorphic child setters and getters so this doesn't need to be implemented for all types
    def resolve_references(self, type_dict):
        pass


TypeDict = Dict[str, Type]


class PrimitiveType(Type):
    def __init__(self, name: str, primitive: TypingType):
        super().__init__(name)
        self.primitive = primitive

    @classmethod
    def from_json(cls, json_object):
        raise TypeError("Can't create a new primitive type from json")

    def validate(self, value):
        assert isinstance(value, self.primitive)


class EnumType(Type):
    def __init__(self, name: str, options: List[str]):
        super().__init__(name)
        self.options = options

    @classmethod
    def from_json(cls, json_object):
        assert_string_exists(json_object, "name")
        assert_string_list_exists(json_object, "options")

        return EnumType(json_object["name"], json_object["options"])

    def validate(self, value):
        assert any([value == option for option in self.options])


class UnionType(Type):
    def __init__(self, name: str, types: List[Type]):
        super().__init__(name)
        self.types = types

    @classmethod
    def from_json(cls, json_object):
        assert_string_exists(json_object, "name")
        assert_string_list_exists(json_object, "types")

        return UnionType(json_object["name"], [ReferenceType(type_str) for type_str in json_object["types"]])

    def validate(self, value):
        assert [_type.validate(value) for _type in self.types]

    def resolve_references(self, type_dict: TypeDict):
        for i, _type in enumerate(self.types):
            if isinstance(_type, ReferenceType):
                if _type.name in type_dict.keys():
                    self.types[i] = type_dict[_type.name]
                else:
                    raise UnknownReferenceError()


class OptionalType(Type):
    def __init__(self, _type: Type):
        super().__init__("optional[%s]" % _type.name)
        self.type = _type

    @classmethod
    def from_json(cls, json_object):
        assert_string_exists(json_object, "type")

        return OptionalType(ReferenceType(json_object["type"]))

    def validate(self, value):
        if value is not None:
            self.type.validate(value)

    def resolve_references(self, type_dict: TypeDict):
        if isinstance(self.type, ReferenceType):
            if self.type.name in type_dict.keys():
                self.type = type_dict[self.type.name]
            else:
                raise UnknownReferenceError()


class ObjectType(Type):
    def __init__(self, name: str, fields: Dict[str, Type]):
        super().__init__(name)
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

    def resolve_references(self, type_dict: TypeDict):
        for key in self.fields.keys():
            _type = self.fields[key]
            if isinstance(_type, ReferenceType):
                if _type.name in type_dict.keys():
                    self.fields[key] = type_dict[_type.name]
                else:
                    raise UnknownReferenceError()


class ReferenceType(Type):
    def __init__(self, name: str):
        super().__init__(name)

    @classmethod
    def from_json(cls, json_object):
        raise TypeError("Can't create a new reference type from json")

    def validate(self, value):
        raise TypeError("Can't validate reference type, must resolve all references first")

    def resolve_references(self, type_dict: TypeDict):
        raise TypeError("Can't resolve references on a reference type")


class UnknownReferenceError(RuntimeError):
    pass


class Instance:
    def __init__(self, value: Any, _type: Type):
        _type.validate(value)

        self.value = value
        self.type = _type


int_type = PrimitiveType("int", int)
float_type = PrimitiveType("float", float)
bool_type = PrimitiveType("bool", bool)


def resolve_references(types: List[Type]):
    types.extend([int_type, float_type, bool_type])

    type_dict = dict((_type.name, _type) for _type in types)

    for _type in types:
        _type.resolve_references(type_dict)


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
