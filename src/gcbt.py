"""Guidance-Correlation Branch Transform, closed form on matched pairs."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_GAP_TOL = 1e-10
_REDUCE_EPS = 1e-15


def guidance(s):
    """Return g_s = [s, 1 - s]."""
    s = jnp.asarray(s, dtype=jnp.float64)
    return jnp.stack([s, jnp.asarray(1.0, dtype=jnp.float64) - s])


def rotation(theta):
    """2x2 branch rotation R(theta) = [[cos, sin], [-sin, cos]]."""
    theta = jnp.asarray(theta, dtype=jnp.float64)
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[c, s], [-s, c]], dtype=jnp.float64)


def _symmetrize(sigma):
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    return 0.5 * (sigma + sigma.T)


def _reduce_mod_half_turn(theta):
    """Reduce modulo pi/2 into (-pi/4, pi/4]."""
    period = jnp.pi / 2.0
    shifted = jnp.mod(theta + jnp.pi / 4.0, period)
    out = shifted - jnp.pi / 4.0
    # jnp.mod lands in [0, period), so out lands in [-pi/4, pi/4).
    # The closed endpoint of the target interval is +pi/4, not -pi/4.
    return jnp.where(out <= -jnp.pi / 4.0 + _REDUCE_EPS, out + period, out)


def _orient_dominant(v):
    """Flip v so its first nonzero entry is positive."""
    # Numerical dust in the leading entry must not steal the sign from a
    # genuinely nonzero later entry (coordinate eigenvectors).
    scale = jnp.maximum(jnp.linalg.norm(v), 1.0)
    leading_real = jnp.abs(v[0]) > 1e-12 * scale
    ref = jnp.where(leading_real, v[0], v[1])
    sign = jnp.where(ref < 0.0, -1.0, 1.0)
    return v * sign


def gcbt_angle(sigma, s):
    """Closed-form branch angle and an isotropic flag.

    Returns theta in (-pi/4, pi/4] and isotropic=True when the eigenvalues
    of Sigma differ by less than 1e-10 * (1 + trace(Sigma)).
    """
    sigma = _symmetrize(sigma)
    evals, evecs = jnp.linalg.eigh(sigma)
    gap = evals[-1] - evals[0]
    thresh = _GAP_TOL * (1.0 + jnp.trace(sigma))
    isotropic = gap < thresh

    v = _orient_dominant(evecs[:, -1])
    phi = jnp.atan2(1.0 - jnp.asarray(s, dtype=jnp.float64), jnp.asarray(s, dtype=jnp.float64))
    psi = jnp.atan2(v[1], v[0])
    theta = (phi + psi) / 2.0 - jnp.pi / 4.0
    theta = _reduce_mod_half_turn(theta)
    theta = jnp.where(isotropic, jnp.asarray(0.0, dtype=jnp.float64), theta)
    return theta, isotropic


def surrogate(theta, sigma, s):
    """Equal-rate quantization-noise surrogate D_sur(theta)."""
    sigma = _symmetrize(sigma)
    rows = rotation(theta)
    g = guidance(s)
    projected = rows @ g
    power = jnp.diag(rows @ sigma @ rows.T)
    return jnp.sum(projected ** 2 * power)


def apply_branch_rotation(x, theta):
    """Z = R(theta) @ X for X shaped (2,) or (2, d)."""
    x = jnp.asarray(x, dtype=jnp.float64)
    return rotation(theta) @ x


def invert_branch_rotation(z, theta):
    """X = R(theta).T @ Z for Z shaped (2,) or (2, d)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    return rotation(theta).T @ z


def quantize_roundtrip(x, theta, quantize_fn):
    """Rotate, quantize in the branch basis, rotate back."""
    return invert_branch_rotation(quantize_fn(apply_branch_rotation(x, theta)), theta)


def estimate_sigma(pairs):
    """Uncentered second moment (1/N) sum_n x_n x_n^T. Does not subtract the mean."""
    pairs = jnp.asarray(pairs, dtype=jnp.float64)
    n = pairs.shape[0]
    return (pairs.T @ pairs) / n
