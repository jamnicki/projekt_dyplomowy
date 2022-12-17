import numpy as np
import unittest

from abc import ABC
from typing import Iterable, Union, Generator, NewType

Idx = NewType("Idx", int)
Score = NewType("Score", Union[float, int])


class Sampler(ABC):
    """Abstract base class for all samplers."""


class LeastConfidenceSampler(Sampler):
    """Least confidence sampling strategy based on given scores."""

    def __init__(self) -> None:
        super().__init__()
        self.indexes_by_score: Iterable[Idx] = []
        self.include_arr: np.ndarray = np.array([])

    def __call__(
        self,
        scores: Iterable[Score],
        exclude: Iterable[Idx],
        num: int,
        strict: bool = True
    ) -> Generator[Union[Idx, None], None, None]:

        scores_len = len(scores)
        if strict and scores_len - len(exclude) < num:
            raise ValueError("Number of samples cannot be larger than lenght "
                             "of the included scores!")
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        self.include_arr = np.repeat(True, scores_len)
        for ex in exclude:
            self.include_arr[ex] = False
        # ascending order
        self.indexes_by_score = np.argsort(scores, kind="stable")
        n_queried = 0
        while n_queried < num:
            idx = self._get_least_confident()
            self.include_arr[idx] = False
            n_queried += 1
            yield idx

    def _get_least_confident(self) -> Union[Idx, None]:
        """Get the index of the least confident score if not excluded."""
        for _idx in self.indexes_by_score:
            include = self.include_arr[_idx]
            if include:
                return _idx


class TestLeastConfidenceSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = LeastConfidenceSampler()

    def test_get_least_confident(self):
        self.sampler.indexes_by_score = [3, 1, 2]
        self.sampler.include_arr = np.array([True, True, True])

        least_confident = self.sampler._get_least_confident()
        self.assertEqual(least_confident, 1)

        self.sampler.include_arr[1] = False
        least_confident = self.sampler._get_least_confident()
        self.assertEqual(least_confident, 2)

        self.sampler.include_arr[2] = False
        least_confident = self.sampler._get_least_confident()
        self.assertEqual(least_confident, 3)

    def test_out_of_index(self):
        self.sampler.include_arr[3] = False
        with self.assertRaises(IndexError):
            self.sampler._get_least_confident()

    def test_all_zeros(self):
        dummy_data = [
            {"spans": {"scores": np.array([0.0])}},
            {"spans": {"scores": np.array([0.0])}},
            {"spans": {"scores": np.array([0.0, 0.0])}},
        ]
        scores = [
            data["spans"]["scores"].mean() if data["spans"] else 0.0
            for data in dummy_data
        ]
        samples_idxs = list(self.sampler(scores, exclude=set(), num=3))
        self.assertSequenceEqual(samples_idxs, [0, 1, 2])

    def test_call_without_exclude(self):
        scores = [0.4, 0.7, 0.9, 0.2]
        exclude = []
        num = 3
        strict = True

        least_confident_samples = self.sampler(scores, exclude, num, strict)
        with self.assertRaises(ValueError):
            self.assertEqual(list(least_confident_samples), [3, 0, 1])

    def test_call_with_exclude(self):
        scores = [0.4, 0.7, 0.9, 0.2]
        exclude = [0, 3]
        num = 3
        strict = True

        least_confident_samples = self.sampler(scores, exclude, num, strict)
        self.assertEqual(list(least_confident_samples), [1, 2])


if __name__ == "__main__":
    unittest.main()
