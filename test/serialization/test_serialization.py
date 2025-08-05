import hashlib

import pytest

from genrl.serialization.game_tree import from_bytes, to_bytes


@pytest.mark.parametrize(
    "obj",
    [
        [1, 2, 3, [5, 6]],
        {"A": ["this", "is", "a cat"], "B": ["this", "is", "a cat", "and dog"]},
        {
            "A": {"there is data": "here"},
            "B": {"this": {"is": {"nested": "dictionary"}}},
        },
    ],
)
def test_to_and_from_bytes(obj):
    serialized_obj = to_bytes(obj)
    deserialized_obj = from_bytes(serialized_obj)
    assert deserialized_obj == obj


def test_to_and_from_bytes_with_hashes():
    hash_object = hashlib.sha256()
    hash_object.update(b"123sdfsdf")
    obj = int(hash_object.hexdigest(), 16)
    serialized_obj = to_bytes(obj)
    deserialized_obj = from_bytes(serialized_obj)
    assert deserialized_obj == obj
