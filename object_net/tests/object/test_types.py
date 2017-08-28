from object_net.object import types
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
