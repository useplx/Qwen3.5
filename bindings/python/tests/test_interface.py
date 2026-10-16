import struct

from qwen35 import Input


def test_text_input_matches_native_contract() -> None:
    payload = Input.text("hello").encode()
    assert struct.unpack_from("<II", payload) == (0x51333549, 1)
    assert struct.unpack_from("<BI", payload, 8) == (0, 5)
    assert payload[13:] == b"hello"
