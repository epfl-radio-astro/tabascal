from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional

from jax import Array


class Component(ABC):
    """Base class for all tabascal components"""

    # Class attributes defining component interface
    required_inputs: Dict[str, tuple] = {}
    outputs: Dict[str, tuple] = {}
    state_outputs: Dict[str, Array] = {}
    init_params: Dict[str, Array] = {}
    init_params_base: Dict[str, Array] = {}

    # parameters: Dict[str, tuple] = {}

    @abstractmethod
    def setup(self, config: Any) -> None:
        """Initialize component with configuration"""
        pass

    @abstractmethod
    def build_forward(self) -> Callable:
        """Build the forward computation function"""

        def forward(state: Dict) -> Dict:
            return state

        return forward

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
