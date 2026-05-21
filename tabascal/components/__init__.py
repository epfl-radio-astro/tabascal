from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Tuple

from jax import Array

class Component(ABC):
    """Base class for all tabascal components"""

    # Class attributes defining component interface
    required_inputs: Dict[str, tuple] = {}
    parameter_shapes: Dict[str, tuple] = {}
    output_shapes: Dict[str, tuple] = {}
    outputs: Dict[str, Array] = {}
    init_params: Dict[str, Array] = {}
    init_params_base: Dict[str, Array] = {}

    # Explicit declaration of the component's state I/O, used by the config-time
    # dependency resolver (``tabascal.config.validate_component_dependencies``).
    # These describe what the ``forward`` built by ``build_forward`` does to the
    # state dict and are the authoritative interface for ordering validation
    # (the legacy ``required_inputs`` / ``output_shape(s)`` attrs are kept for
    # shape metadata but are inconsistent across components and not relied on
    # here).
    #
    # Each maps a state key -> its (symbolic) shape. Dimensions are dim-name
    # strings (e.g. "n_bl", "n_freq") or literal ints; the resolver resolves the
    # names it knows to concrete sizes from the config and compares shapes across
    # the producer->consumer edge.
    #
    # - ``reads``       : state keys consumed but not produced (pure inputs).
    # - ``writes``      : state keys established/overwritten (value does not
    #                     depend on a prior value of the key).
    # - ``accumulates`` : state keys read-modified-written (``state[k] += dk``).
    #                     The key must already be available (seeded by the Model
    #                     or written upstream); contributions are additive.
    reads: Dict[str, Tuple] = {}
    writes: Dict[str, Tuple] = {}
    accumulates: Dict[str, Tuple] = {}

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

    def _set_outputs(self):
        pass


def assert_attr_shape(obj, attr, shape):

    assert hasattr(obj, attr), f"{attr} does not exist."
    attr_shape = getattr(obj, attr).shape
    assert (
        attr_shape == shape
    ), f"Expected shape {shape} for {attr} but got {attr_shape}."
