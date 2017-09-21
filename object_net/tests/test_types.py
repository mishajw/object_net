from object_net import types
import unittest


class TestTypes(unittest.TestCase):
    def test_valid_instance(self):
        tree_types = TestTypes.get_tree_types()

        types.resolve_references(tree_types)

        types.Instance(
            {
                "value": 4,
                "mod_three": "one",
                "left": {
                    "value": 2,
                    "mod_three": "two"
                },
                "right": {
                    "value": 2,
                    "mod_three": "two"
                }
            },
            tree_types[0])

    def test_create_enum(self):
        created_types = types.create_from_json(
            """
            {
                "types": [
                    {
                        "name": "mod_three",
                        "base": "enum",
                        "options": ["one", "two", "three"]
                    }
                ]
            }
            """)

        mod_three = created_types[0]

        assert isinstance(mod_three, types.EnumType)
        self.assertIsInstance(mod_three, types.EnumType)
        self.assertEqual(mod_three.name, "mod_three")
        self.assertCountEqual(mod_three.options, ["one", "two", "three"])

    def test_create_union(self):
        created_types = types.create_from_json(
            """
            {
                "types": [
                    {
                        "name": "int_or_float",
                        "base": "union",
                        "types": ["int", "float"]
                    }
                ]
            }
            """)

        int_or_float = created_types[0]

        assert isinstance(int_or_float, types.UnionType)
        self.assertIsInstance(int_or_float, types.UnionType)
        self.assertEqual(int_or_float.name, "int_or_float")
        self.assertCountEqual([_type.name for _type in int_or_float.types], ["int", "float"])

    def test_create_optional(self):
        created_types = types.create_from_json(
            """
            {
                "types": [
                    {
                        "base": "optional",
                        "type": "int"
                    }
                ]
            }
            """)

        int_opt = created_types[0]

        assert isinstance(int_opt, types.OptionalType)
        self.assertIsInstance(int_opt, types.OptionalType)
        self.assertEqual(int_opt.name, "optional[int]")
        self.assertCountEqual(int_opt.type.name, "int")

    def test_create_object(self):
        created_types = types.create_from_json(
            """
            {
                "types": [
                    {
                        "name": "coordinates",
                        "base": "object",
                        "x": "int",
                        "y": "int"
                    }
                ]
            }
            """)

        coordinates = created_types[0]

        assert isinstance(coordinates, types.ObjectType)
        self.assertIsInstance(coordinates, types.ObjectType)
        self.assertEqual(coordinates.name, "coordinates")

        field_names = coordinates.fields.keys()
        field_types = [coordinates.fields[field_name].name for field_name in field_names]

        self.assertCountEqual(field_names, ["x", "y"])
        self.assertCountEqual(field_types, ["int", "int"])

    def test_resolve_references(self):
        created_types = list(types.create_from_json(
            """
            {
                "types": [
                    {
                        "base": "object",
                        "name": "tree",
                        "value": "int",
                        "mod_three": "mod_three",
                        "left": "optional[tree]",
                        "right": "optional[tree]"
                    },
                    {
                        "base": "enum",
                        "name": "mod_three",
                        "options": ["one", "two", "three"]
                    },
                    {
                        "base": "optional",
                        "type": "tree"
                    }
                ]
            }
            """))

        types.resolve_references(created_types)

        all_types = created_types[0].get_all_types()

        for _type in all_types:
            self.assertNotIsInstance(_type, types.ReferenceType)

    def test_get_all_types(self):
        tree_types = TestTypes.get_tree_types()
        all_types = types.get_all_types(tree_types[0])

        for _type in tree_types:
            self.assertIn(_type, all_types)

    @staticmethod
    def get_tree_types():
        mod_three = types.EnumType("mod_three", ["one", "two", "three"])
        tree_opt = types.OptionalType(types.ReferenceType("tree"))

        tree = types.ObjectType(
            "tree",
            {
                "value": types.int_type,
                "mod_three": mod_three,
                "left": tree_opt,
                "right": tree_opt
            })

        return [tree, mod_three, tree_opt]
