from jax import jit, lax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from numpyro.distributions.transforms import Transform
from numpyro.distributions import constraints


@jit
def affine_transform_full(x, L, mu):
    return L @ x + mu


@jit
def affine_transform_full_inv(x, L_inv, mu):
    return L_inv @ (x - mu)


@jit
def affine_transform_diag(x, sigma, mu):
    return sigma * x + mu


@jit
def affine_transform_diag_inv(x, sigma_inv, mu):
    return sigma_inv * (x - mu)


class LowerCholeskyAffine(Transform):
    r"""
    Transform via the mapping :math:`y = loc + scale\_tril\ @\ x`.

    :param loc: a real vector.
    :param scale_tril: a lower triangular matrix with positive diagonal.

    **Example**

    .. doctest::

       >>> import jax.numpy as jnp
       >>> from numpyro.distributions.transforms import LowerCholeskyAffine
       >>> base = jnp.ones(2)
       >>> loc = jnp.zeros(2)
       >>> scale_tril = jnp.array([[0.3, 0.0], [1.0, 0.5]])
       >>> affine = LowerCholeskyAffine(loc=loc, scale_tril=scale_tril)
       >>> affine(base)
       Array([0.3, 1.5], dtype=float32)
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, loc, scale_tril):
        if jnp.ndim(scale_tril) != 2:
            raise ValueError(
                "Only support 2-dimensional scale_tril matrix. "
                "Please make a feature request if you need to "
                "use this transform with batched scale_tril."
            )
        self.loc = loc
        self.scale_tril = scale_tril

    def __call__(self, x):
        return self.loc + jnp.squeeze(
            jnp.matmul(self.scale_tril, x[..., jnp.newaxis]), axis=-1
        )

    def _inverse(self, y):
        y = y - self.loc
        original_shape = jnp.shape(y)
        yt = jnp.reshape(y, (-1, original_shape[-1])).T
        xt = solve_triangular(self.scale_tril, yt, lower=True)
        return jnp.reshape(xt.T, original_shape)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.broadcast_to(
            jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)).sum(-1),
            jnp.shape(x)[:-1],
        )

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return lax.broadcast_shapes(shape, self.loc.shape, self.scale_tril.shape[:-1])

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return lax.broadcast_shapes(shape, self.loc.shape, self.scale_tril.shape[:-1])

    def tree_flatten(self):
        return (self.loc, self.scale_tril), (("loc", "scale_tril"), dict())

    def __eq__(self, other):
        if not isinstance(other, LowerCholeskyAffine):
            return False
        return jnp.array_equal(self.loc, other.loc) & jnp.array_equal(
            self.scale_tril, other.scale_tril
        )
