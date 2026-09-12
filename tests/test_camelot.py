"""Camelot wheel identities and harmonic distances."""

import pytest

from automix.model.camelot import (
    Key, compatible_keys, from_pitch_class, harmonic_distance, parse, wheel_distance,
)


@pytest.mark.parametrize("pitch_class,is_major,code", [
    (0, True, "8B"), (7, True, "9B"), (2, True, "10B"), (11, True, "1B"),
    (6, True, "2B"), (5, True, "7B"), (9, False, "8A"), (4, False, "9A"),
    (0, False, "5A"), (6, False, "11A"),
])
def test_known_camelot_codes(pitch_class, is_major, code):
    assert from_pitch_class(pitch_class, is_major).code == code


def test_every_key_roundtrips_through_pitch_class():
    for number in range(1, 13):
        for is_major in (True, False):
            key = Key(number, is_major)
            assert from_pitch_class(key.pitch_class, is_major) == key


def test_relative_major_and_minor_share_a_number():
    assert from_pitch_class(0, True).number == from_pitch_class(9, False).number


def test_transposing_a_semitone_moves_seven_steps():
    assert parse("8B").transposed(1).code == "3B"
    assert parse("8B").transposed(12) == parse("8B")
    assert parse("8B").transposed(-1).code == "1B"


def test_distance_ordering_matches_mixing_practice():
    same = harmonic_distance(parse("8A"), parse("8A"))[0]
    adjacent = harmonic_distance(parse("8A"), parse("9A"))[0]
    relative = harmonic_distance(parse("8A"), parse("8B"))[0]
    diagonal = harmonic_distance(parse("8A"), parse("9B"))[0]
    clash = harmonic_distance(parse("8A"), parse("2B"))[0]
    assert same == 0
    assert same < adjacent <= relative < diagonal < clash


def test_wheel_distance_wraps_the_short_way():
    assert wheel_distance(parse("1A"), parse("12A")) == 1
    assert wheel_distance(parse("1A"), parse("7A")) == 6


def test_distance_is_symmetric_on_the_wheel():
    for n in range(1, 13):
        a, b = parse("8A"), Key(n, False)
        assert wheel_distance(a, b) == wheel_distance(b, a)


def test_compatible_keys_puts_the_same_key_first():
    assert compatible_keys(parse("8A"))[0] == parse("8A")


def test_parse_rejects_nonsense():
    for bad in ("", "13A", "8C", "A8"):
        with pytest.raises(ValueError):
            parse(bad)
