import math
import random
import pytest
from vectra.metrics import euclidean_distance
from vectra.pq import ProductQuantizer


def test_pq_fit_and_encode_decode():
    dim = 8
    num_subvectors = 4
    num_centroids = 4

    pq = ProductQuantizer(dimension=dim, num_subvectors=num_subvectors, num_centroids=num_centroids, seed=42)

    # Generate synthetic training vectors
    rng = random.Random(99)
    training = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(30)]

    pq.fit(training, max_iters=5)
    assert pq.is_fitted is True
    assert len(pq.codebooks) == num_subvectors
    assert len(pq.codebooks[0]) == num_centroids

    # Encode a test vector
    test_vec = training[0]
    codes = pq.encode(test_vec)
    assert len(codes) == num_subvectors
    for c in codes:
        assert 0 <= c < num_centroids

    # Decode
    recon = pq.decode(codes)
    assert len(recon) == dim

    # Reconstruction error should be reasonable
    error = euclidean_distance(test_vec, recon)
    assert error < 5.0


def test_pq_asymmetric_distance_computation():
    dim = 4
    pq = ProductQuantizer(dimension=dim, num_subvectors=2, num_centroids=4, seed=42)

    data = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    pq.fit(data, max_iters=3)

    query = [1.0, 0.1, 0.0, 0.0]
    table = pq.compute_distance_table(query)
    assert len(table) == 2

    code_0 = pq.encode(data[0])
    adc_dist = pq.asymmetric_distance_from_table(table, code_0)
    assert adc_dist >= 0.0


def test_pq_dimension_check():
    with pytest.raises(ValueError, match="must be divisible"):
        ProductQuantizer(dimension=7, num_subvectors=4)
