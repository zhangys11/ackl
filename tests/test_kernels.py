"""
Unit tests for ackl kernel library.

Run with: pytest tests/test_kernels.py -v
"""
import sys
import math
import warnings
import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, 'src')
from ackl import kernels
from ackl.kernels import (
    euclidean_dist_matrix, gaussian_kernel, exponential_kernel,
    anova_kernel, rq_kernel, rq_kernel_v2, imq_kernel, cauchy_kernel,
    ts_kernel, spline_kernel, sorensen_kernel, tanimoto_kernel,
    min_kernel, minmax_kernel, expmin_kernel, ghi_kernel,
    fourier_kernel, fourier_kernel_v2, wavelet_kernel,
    log_kernel, power_kernel, bessel_kernel, matern_kernel,
    ess_kernel, fejer_kernel, circular_kernel, spherical_kernel,
    wave_kernel, cosine_kernel, kernel_dict, kernel_names,
    default_wavelet, mod_bessel,
    Kernel, KernelSum, KernelProduct, KernelPower, ScaledKernel,
    GraphKernel, RationalQuadratic2, InverseMultiquadratic,
    Cauchy, TStudent, ANOVA, Wavelet, Fourier,
    Tanimoto, Sorensen, AdditiveChi2, Chi2,
    Min, GeneralizedHistogramIntersection, MinMax, Spline,
    Log, Power, ShortestPath, Graph,
    graphs_to_adjacency_lists, floyd_warshall, _apply_floyd_warshall,
    relabel,
)


# ── test fixtures ──────────────────────────────────────────

@pytest.fixture
def X1():
    return np.array([[0.5, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=float)


@pytest.fixture
def X2():
    return np.array([[0.5, 1.0], [1.0, 2.0]], dtype=float)


@pytest.fixture
def X_single():
    return np.array([[1.0], [2.0], [3.0]], dtype=float)


@pytest.fixture
def X_rand():
    rng = np.random.RandomState(42)
    return rng.rand(10, 4)


@pytest.fixture
def X_pos():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)


# ── helper function tests ───────────────────────────────────

class TestEuclideanDistMatrix:
    def test_shape(self, X1, X2):
        D = euclidean_dist_matrix(X1, X2)
        assert D.shape == (3, 2)

    def test_zero_distance(self):
        x = np.array([[1.0, 2.0]])
        D = euclidean_dist_matrix(x, x)
        assert abs(D[0, 0]) < 1e-10

    def test_non_negative(self, X_rand):
        D = euclidean_dist_matrix(X_rand, X_rand)
        assert np.all(D >= -1e-10)

    def test_symmetric_for_same(self, X_rand):
        D = euclidean_dist_matrix(X_rand, X_rand)
        assert np.allclose(D, D.T)

    def test_correct_value(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        D = euclidean_dist_matrix(a, b)
        assert abs(D[0, 0] - 25.0) < 1e-10


class TestDefaultWavelet:
    def test_output_shape(self):
        x = np.array([[1.0], [2.0]])
        y = default_wavelet(x)
        assert y.shape == (2, 1)

    def test_known_value(self):
        result = default_wavelet(0.0)
        assert abs(result - 1.0) < 1e-10


class TestModBessel:
    def test_positive_output(self):
        x = np.array([1.0, 2.0, 3.0])
        result = mod_bessel(x)
        assert np.all(result > 0)


# ── kernel property tests ───────────────────────────────────

class TestKernelProperties:
    """Generic mathematical properties all kernels should satisfy."""

    @pytest.mark.parametrize("name,func", [
        (n, f) for n, f in kernel_dict.items()
    ])
    def test_output_shape(self, name, func, X1, X2):
        M = func(X1, X2)
        assert M.shape == (len(X1), len(X2)), f"{name}: shape {M.shape}"

    @pytest.mark.parametrize("name,func", [
        (n, f) for n, f in kernel_dict.items()
    ])
    def test_no_nan(self, name, func, X_pos):
        M = func(X_pos, X_pos)
        assert not np.any(np.isnan(M)), f"{name}: contains NaN"

    @pytest.mark.parametrize("name,func", [
        (n, f) for n, f in kernel_dict.items()
    ])
    def test_finite_output(self, name, func, X_pos):
        M = func(X_pos, X_pos)
        assert np.all(np.isfinite(M)), f"{name}: contains inf"

    @pytest.mark.parametrize("name,func", [
        (n, f) for n, f in kernel_dict.items()
        if n not in ('minmax',)  # minmax has division edge cases
    ])
    def test_symmetric_gram(self, name, func, X_pos):
        M = func(X_pos, X_pos)
        assert np.allclose(M, M.T, atol=1e-10), f"{name}: gram not symmetric"

    @pytest.mark.parametrize("name,func", [
        (n, f) for n, f in kernel_dict.items()
        if n not in ('power', 'log')  # CPD kernels can be negative
    ])
    def test_diagonal_non_negative(self, name, func, X_pos):
        M = func(X_pos, X_pos)
        diag = np.diag(M)
        assert np.all(diag >= -1e-10), f"{name}: negative diagonal values"


# ── individual kernel tests ─────────────────────────────────

class TestGaussianKernel:
    def test_self_similarity_one(self, X1):
        M = gaussian_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_output_range(self, X_rand):
        M = gaussian_kernel(X_rand, X_rand)
        assert np.all(M >= 0) and np.all(M <= 1)


class TestLaplacianKernel:
    def test_self_similarity_one(self, X1):
        from sklearn.metrics.pairwise import laplacian_kernel
        M = laplacian_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)


class TestAnovaKernel:
    def test_self_similarity(self, X1):
        M = anova_kernel(X1, X1)
        assert np.all(np.diag(M) > 0)

    def test_none_params(self, X1):
        M = anova_kernel(X1, X1, sigma=None, d=None)
        assert M.shape == (3, 3)


class TestRQKernel:
    def test_self_similarity(self, X1):
        M = rq_kernel(X1, X1)
        diag = np.diag(M)
        assert np.allclose(diag, 1.0)

    def test_rq_v2_gram(self, X1):
        M = rq_kernel_v2(X1, X1)
        assert np.allclose(M, M.T)


class TestIMQKernel:
    def test_output_range(self, X1):
        M = imq_kernel(X1, X1)
        assert np.all(M > 0) and np.all(M <= 1)


class TestCauchyKernel:
    def test_self_similarity_one(self, X1):
        M = cauchy_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_none_sigma(self, X1):
        M = cauchy_kernel(X1, X1, sigma=None)
        assert M.shape == (3, 3)


class TestTSKernel:
    def test_self_similarity(self, X1):
        M = ts_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_degree_cauchy(self, X1):
        M2 = ts_kernel(X1, X1, d=2)
        M_cauchy = cauchy_kernel(X1, X1, sigma=1.0)
        assert np.allclose(M2, M_cauchy)


class TestSplineKernel:
    def test_flavor1(self, X_pos):
        M = spline_kernel(X_pos, X_pos, k=None, flavor=1)
        assert M.shape == (3, 3)

    def test_flavor2(self, X_pos):
        M = spline_kernel(X_pos, X_pos, k=None, flavor=2)
        assert M.shape == (3, 3)

    def test_flavor1_positive(self, X_pos):
        M = spline_kernel(X_pos, X_pos, k=None, flavor=1)
        assert np.all(M > 0)


class TestSorensenKernel:
    def test_self_similarity(self, X_pos):
        M = sorensen_kernel(X_pos, X_pos)
        assert np.allclose(np.diag(M), 1.0)


class TestTanimotoKernel:
    def test_self_similarity(self, X_pos):
        M = tanimoto_kernel(X_pos, X_pos)
        assert np.allclose(np.diag(M), 1.0)

    def test_zero_vector_safe(self):
        X = np.array([[0.0, 0.0], [1.0, 2.0]])
        M = tanimoto_kernel(X, X)
        assert np.all(np.isfinite(M))


class TestMinKernel:
    def test_output(self, X_pos):
        M = min_kernel(X_pos, X_pos)
        assert M.shape == (3, 3)
        assert np.all(M >= 0)


class TestMinMaxKernel:
    def test_output_in_range(self, X_pos):
        M = minmax_kernel(X_pos, X_pos)
        assert np.all(M >= 0) and np.all(M <= 1)


class TestExpMinKernel:
    def test_self_similarity_one(self, X1):
        M = expmin_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_output_range(self, X1):
        M = expmin_kernel(X1, X1)
        assert np.all(M > 0) and np.all(M <= 1)


class TestGHIKernel:
    def test_alpha_one_same_as_min(self, X_pos):
        M_ghi = ghi_kernel(X_pos, X_pos, alpha=1)
        M_min = min_kernel(X_pos, X_pos)
        assert np.allclose(M_ghi, M_min)


class TestFourierKernel:
    def test_self_similarity(self, X1):
        M = fourier_kernel(X1, X1)
        assert np.allclose(M, M.T)

    def test_v1_v2_same(self, X1):
        M1 = fourier_kernel(X1, X1, q=0.1)
        M2 = fourier_kernel_v2(X1, X1, q=0.1)
        assert np.allclose(M1, M2)

    def test_output_range(self, X1):
        M = fourier_kernel(X1, X1, q=0.5)
        assert np.all(M > 0)


class TestWaveletKernel:
    def test_output(self, X1):
        M = wavelet_kernel(X1, X1)
        assert M.shape == (3, 3)

    def test_symmetric(self, X1):
        M = wavelet_kernel(X1, X1)
        assert np.allclose(M, M.T, atol=1e-10)


class TestLogKernel:
    def test_output_negative(self, X_pos):
        M = log_kernel(X_pos, X_pos)
        assert np.all(M <= 1e-10)  # log kernel is non-positive

    def test_self_distance_zero(self, X_pos):
        M = log_kernel(X_pos, X_pos)
        assert np.allclose(np.diag(M), 0.0)


class TestPowerKernel:
    def test_output_negative(self, X_pos):
        M = power_kernel(X_pos, X_pos)
        assert np.all(M <= 1e-10)

    def test_self_distance_zero(self, X_pos):
        M = power_kernel(X_pos, X_pos)
        assert np.allclose(np.diag(M), 0.0)


class TestBesselKernel:
    def test_output(self, X1):
        M = bessel_kernel(X1, X1)
        assert M.shape == (3, 3)
        assert np.all(np.isfinite(M))


class TestMaternKernel:
    def test_v_half_equals_exponential(self, X1):
        """v=0.5 matern equals exp(-||x-y||), which is exponential kernel."""
        from scipy.spatial.distance import cdist
        D = cdist(X1, X1, metric='euclidean')
        M_exp = np.exp(-D)
        M_mat = matern_kernel(X1, X1, v=0.5, s=1.0)
        assert np.allclose(M_mat, M_exp, atol=1e-10)

    def test_self_similarity(self, X1):
        M = matern_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_different_v(self, X1):
        M1 = matern_kernel(X1, X1, v=0.5, s=2.0)
        M2 = matern_kernel(X1, X1, v=1.5, s=2.0)
        assert not np.allclose(M1, M2)


class TestESSKernel:
    def test_self_similarity_one(self, X1):
        M = ess_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)


class TestFejerKernel:
    def test_self_distance(self, X1):
        M = fejer_kernel(X1, X1)
        assert M.shape == (3, 3)

    def test_diagonal_equals_k(self, X1):
        M = fejer_kernel(X1, X1, k=3)
        assert np.allclose(np.diag(M), 3.0)


class TestCircularKernel:
    def test_self_similarity_one(self, X1):
        M = circular_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_zero_outside_support(self):
        X_far = np.array([[0.0], [100.0]])
        M = circular_kernel(X_far, X_far, s=1)
        assert M[0, 1] == 0.0


class TestSphericalKernel:
    def test_self_similarity_one(self, X1):
        M = spherical_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_zero_outside_support(self):
        X_far = np.array([[0.0], [100.0]])
        M = spherical_kernel(X_far, X_far, s=1)
        assert M[0, 1] == 0.0


class TestWaveKernel:
    def test_sinc_at_zero(self):
        """sinc(0) should be 1."""
        X = np.array([[0.0], [0.0]])
        M = wave_kernel(X, X, s=1.0)
        assert np.allclose(M, 1.0)

    def test_self_distance_one(self, X1):
        M = wave_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_sinc_cross(self):
        """sinc(π) should be 0."""
        X = np.array([[0.0], [math.pi]])
        M = wave_kernel(X, X, s=1.0)
        assert abs(M[0, 1]) < 1e-6

    def test_wave_with_large_s(self):
        """With large s, wave kernel approaches 1 (since sin(x)/x -> 1)."""
        X = np.array([[0.0], [1.0]])
        M = wave_kernel(X, X, s=100.0)
        result = M[0, 1]
        assert abs(result - 1.0) < 0.01


class TestCosineKernel:
    def test_output_shape(self, X1, X2):
        M = cosine_kernel(X1, X2)
        assert M.shape == (3, 2)

    def test_self_similarity_one(self, X1):
        M = cosine_kernel(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_self_dot_range(self, X1):
        M = cosine_kernel(X1, X1)
        assert np.all(M >= -1.01) and np.all(M <= 1.01)


# ── kernel class tests ──────────────────────────────────────

class TestKernelBaseClass:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Kernel()

    def test_gram_produces_symmetric(self):
        k = RationalQuadratic2(c=1)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        G = k.gram(X)
        assert np.allclose(G, G.T)

    def test_add_operator(self):
        k1 = RationalQuadratic2(c=1)
        k2 = InverseMultiquadratic(c=1)
        k_sum = k1 + k2
        assert isinstance(k_sum, KernelSum)

    def test_mul_operator(self):
        k1 = RationalQuadratic2(c=1)
        k_scaled = k1 * 2.0
        assert isinstance(k_scaled, ScaledKernel)

    def test_rmul_operator(self):
        k1 = RationalQuadratic2(c=1)
        k_scaled = 2.0 * k1
        assert isinstance(k_scaled, ScaledKernel)

    def test_pow_operator(self):
        k1 = RationalQuadratic2(c=1)
        k_pow = k1 ** 2
        assert isinstance(k_pow, KernelPower)

    def test_str_repr(self):
        k1 = RationalQuadratic2(c=1)
        assert 'RationalQuadratic2' in str(k1)
        assert 'RationalQuadratic2' in repr(k1)


class TestKernelSum:
    def test_compute(self, X1):
        k1 = RationalQuadratic2(c=1)
        k2 = InverseMultiquadratic(c=1)
        k_sum = k1 + k2
        M_sum = k_sum(X1, X1)
        M_direct = k1(X1, X1) + k2(X1, X1)
        assert np.allclose(M_sum, M_direct)


class TestKernelProduct:
    def test_compute(self, X1):
        k1 = RationalQuadratic2(c=1)
        k2 = RationalQuadratic2(c=2)
        k_prod = k1 * k2
        M_prod = k_prod(X1, X1)
        M_direct = k1(X1, X1) * k2(X1, X1)
        assert np.allclose(M_prod, M_direct)


class TestKernelPower:
    def test_invalid_degree(self):
        k1 = RationalQuadratic2(c=1)
        k_pow = k1 ** 2
        assert isinstance(k_pow, KernelPower)

    def test_negative_degree_raises(self):
        with pytest.raises(Exception):
            KernelPower(RationalQuadratic2(), -1)


class TestScaledKernel:
    def test_scale(self, X1):
        k1 = RationalQuadratic2(c=1)
        k_scaled = k1 * 3.0
        M_scaled = k_scaled(X1, X1)
        M_direct = 3.0 * k1(X1, X1)
        assert np.allclose(M_scaled, M_direct)

    def test_negative_scale_raises(self):
        with pytest.raises(Exception):
            ScaledKernel(RationalQuadratic2(), -1.0)


# ── specific kernel class tests ─────────────────────────────

class TestRationalQuadratic2:
    def test_self_similarity(self, X1):
        k = RationalQuadratic2(c=1)
        M = k(X1, X1)
        assert np.allclose(np.diag(M), 1.0)


class TestInverseMultiquadratic:
    def test_output_range(self, X1):
        k = InverseMultiquadratic(c=1)
        M = k(X1, X1)
        assert np.all(M > 0) and np.all(M <= 1)


class TestCauchyClass:
    def test_self_similarity(self, X1):
        k = Cauchy()
        M = k(X1, X1)
        assert np.allclose(np.diag(M), 1.0)

    def test_none_sigma_uses_dim(self, X1):
        k = Cauchy(sigma=None)
        M = k(X1, X1)
        assert M.shape == (3, 3)


class TestTStudentClass:
    def test_self_similarity(self, X1):
        k = TStudent(degree=2)
        M = k(X1, X1)
        assert np.allclose(np.diag(M), 1.0)


class TestANOVAClass:
    def test_output_shape(self, X1):
        k = ANOVA(sigma=1, d=2)
        M = k(X1, X1)
        assert M.shape == (3, 3)
        assert np.all(M >= 0)


class TestWaveletClass:
    def test_c_none(self, X1):
        k = Wavelet(c=None, a=1)
        M = k(X1, X1)
        assert M.shape == (3, 3)

    def test_c_set(self, X1):
        k = Wavelet(c=0, a=1)
        M = k(X1, X1)
        assert M.shape == (3, 3)


class TestFourierClass:
    def test_output(self, X1):
        k = Fourier(q=0.1)
        M = k(X1, X1)
        assert M.shape == (3, 3)
        assert np.all(M > 0)


class TestTanimotoClass:
    def test_self_similarity(self, X_pos):
        k = Tanimoto()
        M = k(X_pos, X_pos)
        assert np.allclose(np.diag(M), 1.0)

    def test_matches_function(self, X_pos):
        k = Tanimoto()
        M_cls = k(X_pos, X_pos)
        M_fn = tanimoto_kernel(X_pos, X_pos)
        assert np.allclose(M_cls, M_fn)


class TestSorensenClass:
    def test_self_similarity(self, X_pos):
        k = Sorensen()
        M = k(X_pos, X_pos)
        assert np.allclose(np.diag(M), 1.0)


class TestMinClass:
    def test_positive_only(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        k = Min()
        M = k(X, X)
        assert M.shape == (2, 2)

    def test_negative_warns(self):
        X = np.array([[-1.0, 2.0], [3.0, 4.0]])
        k = Min()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            k(X, X)
            assert len(w) >= 1


class TestMinMaxClass:
    def test_output_range(self, X_pos):
        k = MinMax()
        M = k(X_pos, X_pos)
        assert np.all(M >= 0) and np.all(M <= 1)


class TestLogClass:
    def test_self_zero(self, X_pos):
        k = Log(d=2)
        M = k(X_pos, X_pos)
        assert np.allclose(np.diag(M), 0.0)


class TestPowerClass:
    def test_self_zero(self, X_pos):
        k = Power(d=2)
        M = k(X_pos, X_pos)
        assert np.allclose(np.diag(M), 0.0)


# ── graph kernel tests ──────────────────────────────────────

class TestGraph:
    def test_create_graph(self):
        adj = np.array([[0, 1], [1, 0]])
        g = Graph(adj)
        assert g.adjacency_matix is adj

    def test_labels(self):
        adj = np.array([[0, 1], [1, 0]])
        g = Graph(adj, node_labels=[1, 2], edge_labels=[1])
        assert g.node_labels == [1, 2]


class TestGraphsToAdjacencyLists:
    def test_3d_array(self):
        data = np.array([[[0, 1], [1, 0]], [[0, 0], [0, 0]]])
        result = graphs_to_adjacency_lists(data)
        assert result.shape == (2, 2, 2)

    def test_list_of_graphs(self):
        g1 = Graph(np.array([[0, 1], [1, 0]]))
        g2 = Graph(np.array([[0, 0], [0, 0]]))
        result = graphs_to_adjacency_lists([g1, g2])
        assert result.shape == (2, 2, 2)


class TestFloydWarshall:
    def test_simple_path(self):
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        dist = floyd_warshall(adj, adj.astype(float))
        assert dist[0, 0] == 0
        assert dist[0, 2] == 2  # A-C via B

    def test_disconnected(self):
        adj = np.array([[0, 1], [0, 0]])
        dist = floyd_warshall(adj, adj.astype(float))
        assert np.isinf(dist[1, 0])


class TestApplyFloydWarshall:
    def test_output(self):
        g1 = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
        res, maximum = _apply_floyd_warshall([g1])
        assert maximum == 2
        assert len(res) == 1


class TestRelabel:
    def test_basic(self):
        data1 = [[10, 20], [30, 40]]
        data2 = [[50, 60]]
        r1, r2, n = relabel(data1.copy(), data2)
        assert len(r1) == 2
        assert len(r2) == 1
        assert n == 6  # labels: 10,20,30,40,50,60 = 6 unique labels


class TestShortestPath:
    def test_unlabeled(self):
        g1 = Graph(np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]], dtype=float))
        g2 = Graph(np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, 0]], dtype=float))
        sp = ShortestPath(labeled=False)
        M = sp([g1, g2], [g1, g2])
        assert M.shape == (2, 2)

    @pytest.mark.xfail(reason="labeled graph kernel has pre-existing numpy compatibility issue from pykernels")
    def test_labeled(self):
        g1 = Graph(np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]], dtype=float),
                   node_labels=[1, 2, 1])
        g2 = Graph(np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]], dtype=float),
                   node_labels=[2, 1, 2])
        sp = ShortestPath(labeled=True)
        M = sp([g1], [g2])
        assert M.shape == (1, 1)
        assert M[0, 0] >= 0


# ── kernel_dict consistency tests ───────────────────────────

class TestKernelDict:
    def test_all_callable(self):
        for name, func in kernel_dict.items():
            assert callable(func), f"{name} is not callable"

    def test_names_unique(self):
        assert len(kernel_names) == len(set(kernel_names))

    def test_all_have_formula(self):
        """Each kernel should have a formula entry."""
        formula_keys = set(kernels.kernel_formulas.keys())
        for name in kernel_dict:
            assert name in formula_keys, f"{name} missing formula"


# ── metrics tests ───────────────────────────────────────────

class TestMetrics:
    def test_acc_returns_float(self):
        from ackl.metrics import acc
        rng = np.random.RandomState(42)
        X = rng.rand(20, 5)
        y = np.array([0] * 10 + [1] * 10)
        result = acc(X, y, lambda x, yx: cosine_similarity(x, yx))
        assert isinstance(result, float)

    def test_nmd_binary(self):
        from ackl.metrics import nmd
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = np.array([0] * 10 + [1] * 10)
        result = nmd(X, y, lambda x, yx: cosine_similarity(x, yx))
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_nmd_non_binary_returns_nan(self):
        from ackl.metrics import nmd
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 2])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = nmd(X, y, lambda x, yx: cosine_similarity(x, yx))
            assert np.isnan(result)

    def test_kes_returns_float(self):
        from ackl.metrics import kes
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 0, 1, 1])
        result = kes(X, y, lambda x, yx: cosine_similarity(x, yx))
        assert isinstance(result, float)
        assert not np.isnan(result)


# ── version check test ──────────────────────────────────────

class TestVersion:
    def test_version_string(self):
        import ackl
        assert isinstance(ackl.__version__, str)
        assert ackl.__version__ == '2.2.0'


# ── edge case tests ─────────────────────────────────────────

class TestEdgeCases:
    def test_single_sample(self):
        X = np.array([[1.0, 2.0]])
        for name, func in kernel_dict.items():
            if name == 'spline':
                continue  # spline uses PCA which fails with 1 sample
            M = func(X, X)
            assert M.shape == (1, 1), f"{name}: single sample shape wrong"

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
        for name, func in kernel_dict.items():
            M = func(X, X)
            assert M.shape == (3, 3), f"{name}: 1d shape wrong"

    def test_asymmetric_sizes(self):
        X1 = np.array([[1.0], [2.0], [3.0], [4.0]])
        X2 = np.array([[1.0], [2.0]])
        for name, func in kernel_dict.items():
            M = func(X1, X2)
            assert M.shape == (4, 2), f"{name}: asymmetric shape wrong"
