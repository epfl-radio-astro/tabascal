from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

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

    @abstractmethod
    def setup(self, config: Any) -> None:
        """Initialize component with configuration"""
        pass

    @abstractmethod
    def build_forward(self) -> Callable:
        """Build the forward computation function"""

        def forward(params: Dict, state: Dict) -> Dict:
            return state

        return forward

    def build_constants(self) -> Dict[str, Any]:
        """Return arrays to pass via state instead of closure.

        Returns a dict of array_name -> array_value. These will be stored
        in state as "_c/<ClassName>/array_name" by Model.__init__.
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
