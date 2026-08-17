from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

import jax.numpy as jnp
from jax import Array

from tabascal.config_schema import Param


class Component(ABC):
    """Base class for all tabascal components"""

    # Class attributes defining component interface
    required_inputs: Dict[str, tuple] = {}
    parameter_shapes: Dict[str, tuple] = {}
    output_shapes: Dict[str, tuple] = {}
    outputs: Dict[str, Array] = {}
    init_params: Dict[str, Array] = {}
    init_params_base: Dict[str, Array] = {}

    #: Config parameters this component reads, keyed by their dotted path in the
    #: config file. Declared here, next to the code that reads them, so a
    #: component and its requirements can never drift apart. Only the components
    #: named in ``model.components`` contribute to a run's schema -- see
    #: :func:`tabascal.config_schema.collect_params`. Subclasses extend (and may
    #: override) what their base declared; the parameters read outside any
    #: component live on :class:`tabascal.config.TabConfig`.
    config_params: Dict[str, Param] = {}

    # Set True on components that only work in double precision (read by the
    # run-time preflight in scripts._run_tabascal_impl and by require_double).
    requires_double: bool = False

    @abstractmethod
    def setup(self, tab_config: Any) -> None:
        """Initialize component with configuration"""
        pass

    @abstractmethod
    def build_forward(self) -> Callable:
        """Build the forward computation function"""

        def forward(params: Dict, state: Dict, constants: Dict) -> Dict:
            return state

        return forward

    @property
    def prefix(self) -> str:
        return f"_c/{self.__class__.__name__}"

    def build_constants(self) -> Dict[str, Any]:
        """Return arrays that do not change during the forward pass.

        Returns a dict of array_name -> array_value. These will be stored
        in constants as "_c/<ClassName>/array_name" by Model.__init__.
        """
        return {}

    def build_set_params(self) -> Callable:
        """Build parameter sampling function (optional)"""

        def set_params(params: Dict) -> Dict:

            return params

        return set_params

    def validate_state(self, state: Dict[str, Any]) -> None:
        """Validate required inputs are present"""
        for key in self.required_inputs:
            if key not in state:
                raise ValueError(f"Required input '{key}' missing from state")

    def require_double(self, config: Any) -> None:
        """Raise if this ``requires_double`` component is run in single precision.

        Some components only work in double precision: the SGP4/phase trajectory
        components (differentiable orbits). Those set ``requires_double = True``
        and call this at the top of ``setup`` so they fail with a clear message
        under single precision instead of producing silently-wrong fp32 results.
        Driven by the ``requires_double`` flag so it stays in sync with the
        run-time preflight.
        """
        if self.requires_double and config.precision != "double":
            raise ValueError(
                f"{self.__class__.__name__} requires double precision; "
                "set model.precision to 'double' in the config."
            )

    def _set_outputs(self):
        pass


def axis_extent(x, dx) -> float:
    """The span of a sampled axis, falling back to its sample spacing.

    The default correlation length of a Gaussian-process prior is the extent of
    the axis it is defined on. A single-sample axis has zero extent, which is not
    a usable length scale, so the sample spacing (one channel / one integration)
    stands in for it.
    """
    span = float(jnp.max(x) - jnp.min(x))
    return span if span != 0.0 else float(dx)


def assert_attr_shape(obj, attr, shape):

    assert hasattr(obj, attr), f"{attr} does not exist."
    attr_shape = getattr(obj, attr).shape
    assert (
        attr_shape == shape
    ), f"Expected shape {shape} for {attr} but got {attr_shape}."
