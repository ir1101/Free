"""Camelot wheel: musical key representation and harmonic compatibility.

The Camelot wheel relabels the circle of fifths as 1-12 plus a letter
(A = minor, B = major) so that harmonically compatible keys sit next to each
other. Moving one step around the wheel, or flipping the letter, is a smooth
key change; jumping across it is not.

Layout: 8B = C major, 9B = G major, ... (clockwise by fifths). The A ring holds
the relative minor of the B key with the same number, so 8A = A minor.
"""

from __future__ import annotations

from dataclasses import dataclass

# Pitch classes, C = 0.
PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Transposing up one semitone moves you 7 steps clockwise on the wheel.
SEMITONE_STEP = 7


@dataclass(frozen=True, slots=True)
class Key:
    """A musical key, as both a Camelot code and a conventional name."""

    number: int  # 1-12
    is_major: bool

    @property
    def code(self) -> str:
        return f"{self.number}{'B' if self.is_major else 'A'}"

    @property
    def pitch_class(self) -> int:
        """Tonic pitch class, C = 0."""
        # Invert number = ((pc * 7) % 12 + 7) % 12 + 1 for the major case.
        fifths = (self.number - 8) % 12
        major_pc = (fifths * 7) % 12  # 7 is its own inverse mod 12
        return major_pc if self.is_major else (major_pc - 3) % 12

    @property
    def name(self) -> str:
        return f"{PITCH_NAMES[self.pitch_class]} {'major' if self.is_major else 'minor'}"

    def transposed(self, semitones: int) -> "Key":
        """This key shifted by `semitones` (e.g. from playbackRate tempo matching)."""
        number = (self.number - 1 + SEMITONE_STEP * semitones) % 12 + 1
        return Key(number, self.is_major)

    def __str__(self) -> str:
        return f"{self.code} ({self.name})"


def from_pitch_class(pitch_class: int, is_major: bool) -> Key:
    """Build a Key from a tonic pitch class (C = 0) and mode."""
    # A minor key sits at the same wheel number as its relative major (tonic + 3).
    major_pc = pitch_class % 12 if is_major else (pitch_class + 3) % 12
    number = ((major_pc * 7) % 12 + 7) % 12 + 1
    return Key(number, is_major)


def parse(code: str) -> Key:
    """Parse a Camelot code such as '8A' or '12b'."""
    code = code.strip().upper()
    if len(code) < 2 or code[-1] not in "AB":
        raise ValueError(f"not a Camelot code: {code!r}")
    number = int(code[:-1])
    if not 1 <= number <= 12:
        raise ValueError(f"Camelot number out of range: {code!r}")
    return Key(number, code[-1] == "B")


def wheel_distance(a: Key, b: Key) -> int:
    """Shortest number of steps around the 12-position ring, ignoring the letter."""
    d = abs(a.number - b.number) % 12
    return min(d, 12 - d)


# Cost of each recognised move, roughly ordered by how transparent it sounds in
# a mix. Anything not listed falls through to a distance-scaled penalty.
_PERFECT = 0.0
_ADJACENT = 1.0  # +/-1 around the wheel, or relative major/minor
_DIAGONAL = 2.0  # +/-1 *and* a mode change
_WHOLE_TONE = 2.5  # +2: the classic "energy boost"
_SEMITONE = 3.5  # +7: a deliberate lift, noticeable
_INCOMPATIBLE = 6.0


def harmonic_distance(a: Key, b: Key) -> tuple[float, str]:
    """Cost of moving from key `a` to key `b`, with a human-readable label.

    Lower is smoother. Returned label is shown in the transition inspector so
    it is obvious *why* a pair scored the way it did.
    """
    if a == b:
        return _PERFECT, "same key"

    same_mode = a.is_major == b.is_major
    step = (b.number - a.number) % 12
    dist = wheel_distance(a, b)

    if same_mode:
        if dist == 1:
            return _ADJACENT, f"{'+' if step == 1 else '-'}1 on the wheel"
        if step == 2:
            return _WHOLE_TONE, "+2 energy boost (up a tone)"
        if step == 7:
            return _SEMITONE, "+7 energy boost (up a semitone)"
        if dist == 2:
            return _WHOLE_TONE, "-2 on the wheel (down a tone)"
    else:
        if dist == 0:
            return _ADJACENT, "relative major/minor"
        if dist == 1:
            return _DIAGONAL, "diagonal (one step + mode change)"

    # Everything else: scale with how far apart they are, plus a nudge for a
    # mode change, so the ordering stays sensible rather than flat.
    return _INCOMPATIBLE + dist * 0.5 + (0.0 if same_mode else 0.5), "clashing"


def compatible_keys(a: Key) -> list[Key]:
    """The keys that mix transparently out of `a`, best first. Used by the UI."""
    candidates = [
        Key(a.number, a.is_major),
        Key(a.number, not a.is_major),
        Key(a.number % 12 + 1, a.is_major),
        Key((a.number - 2) % 12 + 1, a.is_major),
    ]
    return sorted(candidates, key=lambda k: harmonic_distance(a, k)[0])
