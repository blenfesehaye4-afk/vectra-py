import math
import random
import pytest
from vectra.metrics import cosine_similarity
from vectra.sq import ScalarQuantizer


def test_sq_fit_and_quantize_bounds():
    sq = ScalarQuantizer(dimension=4)
    data = [
        [0.0, 10.0, -5.0, 1.0],
        [1.0, 20.0, 5.0, 2.0],
        [0.5, 15.0, 0.0, 1.5],
    ]
    sq.fit(data)
    assert sq.is_fitted is True

    q = sq.quantize([0.5, 15.0, 0.0, 1.5])
    assert len(q) == 4
    for val in q:
        assert -128 <= val <= 127


def test_sq_reconstruction_precision():
    dim = 8
    sq = ScalarQuantizer(dimension=dim)
    rng = random.Random(42)

    dataset = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(50)]
    sq.fit(dataset)

    sample = dataset[0]
    q = sq.quantize(sample)
    recon = sq.dequantize(q)

    # Reconstructed vector should correlate strongly with original
    sim = cosine_similarity(sample, recon)
    assert sim > 0.98


def test_sq_fast_cosine_correlation():
    sq = ScalarQuantizer(dimension=3)
    dataset = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.707, 0.707, 0.0],
    ]
    sq.fit(dataset)

    q1 = sq.quantize(dataset[0])
    q2 = sq.quantize(dataset[2])

    approx_sim = sq.fast_cosine(q1, q2)
    true_sim = cosine_similarity(dataset[0], dataset[2])

    assert math.isclose(approx_sim, true_sim, abs_tol=0.1)


def test_sq_not_fitted_error():
    sq = ScalarQuantizer(dimension=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        sq.quantize([1.0, 2.0])
