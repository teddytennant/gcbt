"""CPU checks for the closed-form GCBT angle and the noise surrogate."""

import jax
import jax.numpy as jnp
import pytest

import gcbt

jax.config.update("jax_enable_x64", True)


def _orthogonal(angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.float64)


def _reference_theta(sigma, s):
    evals, evecs = jnp.linalg.eigh(gcbt._symmetrize(sigma))
    v = gcbt._orient_dominant(evecs[:, -1])
    phi = jnp.atan2(1.0 - s, s)
    psi = jnp.atan2(v[1], v[0])
    return (phi + psi) / 2.0 - jnp.pi / 4.0, evals


def _mod_half_turn_distance(a, b):
    period = jnp.pi / 2.0
    delta = jnp.mod((a - b) + jnp.pi / 4.0, period) - jnp.pi / 4.0
    return jnp.where(delta <= -jnp.pi / 4.0 + 1e-15, delta + period, delta)


def test_closed_form_minimizes_surrogate():
    angles = [0.3, 1.1, -0.7, 2.4, 0.0]
    s_values = [0.0, 1.0, 1.5, 7.5, -1.0]
    grid = jnp.linspace(-jnp.pi / 2.0 + 1e-9, jnp.pi / 2.0 - 1e-9, 2000)
    spectrum = jnp.array([3.0, 0.2], dtype=jnp.float64)

    for angle in angles:
        basis = _orthogonal(angle)
        sigma = basis @ jnp.diag(spectrum) @ basis.T
        for s in s_values:
            theta, isotropic = gcbt.gcbt_angle(sigma, s)
            assert bool(isotropic) is False
            th = float(theta)
            assert -jnp.pi / 4.0 - 1e-9 < th <= jnp.pi / 4.0 + 1e-9

            values = jax.vmap(lambda t: gcbt.surrogate(t, sigma, s))(grid)
            d_star = gcbt.surrogate(theta, sigma, s)
            d_min = jnp.min(values)
            rel_gap = (d_star - d_min) / jnp.maximum(jnp.abs(d_min), 1e-12)
            assert float(rel_gap) < 1e-6, (angle, s, float(rel_gap), float(d_star), float(d_min))

            theta_ref, _ = _reference_theta(sigma, s)
            mismatch = abs(float(_mod_half_turn_distance(theta, theta_ref)))
            assert mismatch < 1e-8, (angle, s, mismatch, th, float(theta_ref))


def test_isotropic_surrogate_is_constant():
    s = 1.5
    grid = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, 512, endpoint=False)
    for scale in (0.3, 1.0, 2.5, 10.0):
        sigma = scale * jnp.eye(2, dtype=jnp.float64)
        theta, isotropic = gcbt.gcbt_angle(sigma, s)
        assert bool(isotropic) is True
        assert abs(float(theta)) < 1e-12
        values = jax.vmap(lambda t: gcbt.surrogate(t, sigma, s))(grid)
        assert float(jnp.max(values) - jnp.min(values)) < 1e-8

    # Gap below 1e-10 * (1 + trace) is isotropic; a clearly larger gap is not.
    sigma_near = jnp.diag(jnp.array([1.0, 1.0 + 1e-12]))
    _, iso_near = gcbt.gcbt_angle(sigma_near, s)
    assert bool(iso_near) is True
    sigma_far = jnp.diag(jnp.array([1.0, 1.0 + 1e-6]))
    _, iso_far = gcbt.gcbt_angle(sigma_far, s)
    assert bool(iso_far) is False


def _guidance_mse(samples, theta, sigma, s, kappa, key):
    """Rotate, add independent branch noise, rotate back, MSE along g_s."""
    rows = gcbt.rotation(theta)
    power = jnp.diag(rows @ sigma @ rows.T)
    noise_std = jnp.sqrt(kappa * power)
    z = rows @ samples
    eps = jax.random.normal(key, z.shape, dtype=jnp.float64) * noise_std[:, None]
    recon = rows.T @ (z + eps)
    err = recon - samples
    proj = gcbt.guidance(s) @ err
    return jnp.mean(proj ** 2)


def test_noise_model_matches_surrogate():
    n = 200_000
    kappa = 0.25
    s = 1.7
    basis = _orthogonal(0.7)
    cov = basis @ jnp.diag(jnp.array([3.0, 0.2])) @ basis.T
    key = jax.random.PRNGKey(0)
    key, key_x, key_n1, key_n2 = jax.random.split(key, 4)
    latent = jax.random.normal(key_x, (2, n), dtype=jnp.float64)
    # Zero-mean correlated Gaussian. Population second moment is cov.
    samples = basis @ jnp.diag(jnp.sqrt(jnp.array([3.0, 0.2]))) @ latent
    sigma = cov

    theta_star, isotropic = gcbt.gcbt_angle(sigma, s)
    assert bool(isotropic) is False
    theta_other = float(theta_star) + 0.4

    theory_star = kappa * gcbt.surrogate(theta_star, sigma, s)
    theory_other = kappa * gcbt.surrogate(theta_other, sigma, s)
    emp_star = _guidance_mse(samples, theta_star, sigma, s, kappa, key_n1)
    emp_other = _guidance_mse(samples, theta_other, sigma, s, kappa, key_n2)

    # Chi-square sampling error: sd(mean of squares) = theory * sqrt(2/N).
    tol_star = 6.0 * float(theory_star) * jnp.sqrt(2.0 / n)
    tol_other = 6.0 * float(theory_other) * jnp.sqrt(2.0 / n)
    assert abs(float(emp_star - theory_star)) < tol_star
    assert abs(float(emp_other - theory_other)) < tol_other
    assert float(theory_star) < float(theory_other)
    assert float(emp_star) < float(emp_other)


def test_rotation_commutes_with_feature_map():
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (2, 8), dtype=jnp.float64)
    w = jax.random.normal(k2, (8, 5), dtype=jnp.float64)
    theta = 0.37
    rows = gcbt.rotation(theta)
    left = rows @ (x @ w)
    right = (rows @ x) @ w
    assert jnp.allclose(left, right)
    assert left.shape == (2, 5)

    # Both advertised layouts round-trip.
    vec = x[:, 0]
    assert jnp.allclose(gcbt.invert_branch_rotation(gcbt.apply_branch_rotation(vec, theta), theta), vec)
    assert jnp.allclose(gcbt.invert_branch_rotation(gcbt.apply_branch_rotation(x, theta), theta), x)


def test_quantize_roundtrip_and_surrogate_vs_native_basis():
    # Strongly 1D source, axis aligned with g_s and not a coordinate axis.
    # D_sur models coordinate variance, not uniform rounding. When the signal
    # axis equals the guidance axis the optimum is the bisector, so a
    # round-to-integer quantizer is not the paper's comparison against theta=0.
    s = 1.5
    g = gcbt.guidance(s)
    axis = g / jnp.linalg.norm(g)
    assert abs(float(axis[0])) > 0.2 and abs(float(axis[1])) > 0.2
    perp = jnp.array([-axis[1], axis[0]], dtype=jnp.float64)

    n = 20_000
    key = jax.random.PRNGKey(42)
    k_sig, k_noise = jax.random.split(key)
    coeff = jax.random.normal(k_sig, (n,), dtype=jnp.float64) * 2.0  # variance 4
    noise = jax.random.normal(k_noise, (n,), dtype=jnp.float64) * jnp.sqrt(0.05)
    pairs = coeff[:, None] * axis[None, :] + noise[:, None] * perp[None, :]
    sigma = gcbt.estimate_sigma(pairs)
    theta, isotropic = gcbt.gcbt_angle(sigma, s)
    assert bool(isotropic) is False

    def quantize_fn(z):
        return jnp.round(z)  # step 1.0

    source = pairs.T  # (2, d)
    for angle in (theta, 0.0):
        hat = gcbt.quantize_roundtrip(source, angle, quantize_fn)
        assert hat.shape == source.shape
        assert bool(jnp.all(jnp.isfinite(hat)))

    d_star = float(gcbt.surrogate(theta, sigma, s))
    d_native = float(gcbt.surrogate(0.0, sigma, s))
    assert d_star < d_native, (d_star, d_native)


def test_estimate_sigma_is_uncentered():
    pairs = jnp.array(
        [
            [1.0, 2.0],
            [3.0, -1.0],
            [0.5, 4.0],
            [-2.0, 0.5],
        ],
        dtype=jnp.float64,
    )
    manual = jnp.zeros((2, 2), dtype=jnp.float64)
    for row in pairs:
        manual = manual + jnp.outer(row, row)
    manual = manual / pairs.shape[0]
    sigma = gcbt.estimate_sigma(pairs)
    assert jnp.allclose(sigma, manual)

    mean = jnp.mean(pairs, axis=0)
    assert float(jnp.linalg.norm(mean)) > 0.1
    centered = pairs - mean
    sigma_centered = (centered.T @ centered) / pairs.shape[0]
    assert not jnp.allclose(sigma, sigma_centered, atol=1e-6)

    # A constant nonzero pair must keep the outer product, not collapse to 0.
    constant = jnp.array([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    assert jnp.allclose(gcbt.estimate_sigma(constant), jnp.array([[1.0, 0.0], [0.0, 0.0]]))


def test_guidance_vector():
    g = gcbt.guidance(1.5)
    assert g.shape == (2,)
    assert jnp.allclose(g, jnp.array([1.5, -0.5]))
