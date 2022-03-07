"""Test if random number generator produces same results for same seeds."""
import numpy as np
from bootstrap import get_rng


def test_rng():
    """Test if random number generator produces same results for same seeds."""
    seed = "test"
    kwargs = dict(low=0, high=100, size=[10, 20])
    # fmt: off
    legacy = np.array([
        [66, 96, 18, 57, 52, 23, 76, 32, 93, 62,  2, 70, 19, 45,  6, 66, 66, 38, 70, 63],
        [25, 93, 20, 39, 96, 16, 34, 87, 99, 27, 64, 40, 26, 63, 32, 16, 39, 15, 53, 43],
        [87, 66,  9, 45, 41, 22, 56, 80, 57, 96, 64, 27, 10, 67, 43, 64,  8, 36, 60, 29],
        [90, 32, 98, 22, 32, 12, 89, 57, 87, 97, 39, 98, 27, 92, 65, 76, 43, 69, 66, 23],
        [83, 63, 22,  6, 79, 49, 43, 74, 38, 24, 80, 91, 49, 37, 86,  3, 94, 59, 14,  4],
        [30,  5, 78, 24, 72, 60, 12, 78,  4, 73, 10, 27, 64, 57, 73, 31, 53,  1, 23, 77],
        [61, 27, 25, 19, 26, 18, 70, 12, 90, 54, 76, 86,  0, 48,  2, 25, 87, 93, 84,  1],
        [55, 36, 23, 48, 45, 20, 17, 59, 88, 74,  5,  6, 34, 29, 76,  9, 93, 81, 47, 20],
        [ 7, 96, 11, 42, 79, 96, 59, 40, 76, 96, 52, 53, 90, 12, 49, 37, 88,  6, 23, 28],
        [63, 20, 38, 40, 42, 33, 64, 31, 69, 90, 79, 64, 84, 14, 31, 56, 26, 99, 40, 88]
    ])
    # fmt: on
    rng = get_rng(seed)
    integers = rng.integers(**kwargs)
    np.testing.assert_equal(integers, legacy)
