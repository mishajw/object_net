from typing import Dict, List, Any
from typing import Type as TypingType


class Type:
    def __init__(self, name: str):
        self.name = name

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

    def validate(self, value):
        assert isinstance(value, self.primitive)


class EnumType(Type):
    def __init__(self, name: str, options: List[str]):
        super().__init__(name)
        self.options = options

    def validate(self, value):
        assert any([value == option for option in self.options])


class UnionType(Type):
    def __init__(self, name: str, types: List[Type]):
        super().__init__(name)
        self.types = types

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
