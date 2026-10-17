"""LED-index codebooks for structured-light calibration.

A codebook maps each LED index to a binary codeword. Every codeword is sent
as a run of pattern frames (and, for differential schemes, their complements),
so the decoder can recover each pixel's codeword and therefore the LED's ID.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Codebook:
    name: str
    words: np.ndarray  # (n_leds, n_bits) uint8
    min_distance: int

    @property
    def n_bits(self) -> int:
        return self.words.shape[1]

    @property
    def n_leds(self) -> int:
        return self.words.shape[0]


def _bits(values: np.ndarray, n_bits: int) -> np.ndarray:
    return ((values[:, None] >> np.arange(n_bits)) & 1).astype(np.uint8)


def gray(i: np.ndarray) -> np.ndarray:
    return i ^ (i >> 1)


def min_distance(words: np.ndarray) -> int:
    w = words.astype(np.int16)
    d = (w[:, None, :] != w[None, :, :]).sum(-1)
    np.fill_diagonal(d, 999)
    return int(d.min())


def hamming_secded(data: np.ndarray) -> np.ndarray:
    """Extended Hamming code over the data bits, shortened to fit.

    For 9 data bits this is a (14, 9) code with minimum distance 4: one bit
    error is corrected and two are detected under hard decoding, and soft
    decoding gets a 4-bit margin between codewords.
    """
    k = data.shape[1]
    r = 1
    while (1 << r) < k + r + 1:
        r += 1
    n = k + r
    words = np.zeros((data.shape[0], n), np.uint8)
    data_pos = [p for p in range(1, n + 1) if p & (p - 1)]
    for j, p in enumerate(data_pos):
        words[:, p - 1] = data[:, j]
    for b in range(r):
        p = 1 << b
        cover = [q for q in range(1, n + 1) if q & p and q != p]
        words[:, p - 1] = np.bitwise_xor.reduce(words[:, [q - 1 for q in cover]], axis=1)
    overall = np.bitwise_xor.reduce(words, axis=1)[:, None]
    return np.hstack([words, overall])


def make_codebook(name: str, n_leds: int = 500) -> Codebook:
    """Build a named codebook.

    binary         index in plain binary (the legacy /calibrate/binary scheme)
    gray           reflected Gray code of the index; wire neighbours differ in
                   one bit, so blobs of adjacent LEDs that overlap in the image
                   decode to one of the two rather than to an unrelated LED
    gray_parity    Gray code plus one even-parity bit (detects single errors)
    hamming        extended Hamming (SECDED) over the Gray code, distance 4
    """
    n_bits = max(1, int(np.ceil(np.log2(n_leds))))
    idx = np.arange(n_leds)
    if name == "binary":
        words = _bits(idx, n_bits)
    elif name == "gray":
        words = _bits(gray(idx), n_bits)
    elif name == "gray_parity":
        g = _bits(gray(idx), n_bits)
        words = np.hstack([g, (g.sum(1) & 1)[:, None].astype(np.uint8)])
    elif name == "hamming":
        words = hamming_secded(_bits(gray(idx), n_bits))
    else:
        raise ValueError(f"unknown codebook {name!r}")
    return Codebook(name, words, min_distance(words))


CODEBOOKS = ("binary", "gray", "gray_parity", "hamming")
