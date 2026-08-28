from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Any, Callable, Iterable, List, Sequence
import importlib
import inspect
import pkgutil

from jax import Array


class MissingStateInput(ValueError):
    """A component declares a required input that is not in the model state.

    Carries the component and key so the caller can say *why* it is missing --
    see :func:`validate_component_order`.
    """

    def __init__(self, component: str, key: str, message: str):
        self.component = component
        self.key = key
        super().__init__(message)


class ComponentOrderError(ValueError):
    """``model.components`` is incomplete or in the wrong dependency order."""


class Component(ABC):
    """Base class for all tabascal components"""

    # Class attributes defining component interface
    required_inputs: Dict[str, tuple] = {}
    parameter_shapes: Dict[str, tuple] = {}
    # State keys this component writes. Declared on the class so the component
    # list can be checked before any component is set up; ``setup`` then realises
    # the same keys as ``state_outputs``, which is what Model merges.
    output_shapes: Dict[str, tuple] = {}
    state_outputs: Dict[str, Array] = {}
    outputs: Dict[str, Array] = {}
    init_params: Dict[str, Array] = {}
    init_params_base: Dict[str, Array] = {}

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
        """Raise if a declared required input is not yet in the model state.

        Only the *keys* of ``state`` are meaningful here: this runs at assembly
        time, where the values are placeholders for keys the model supplies
        itself. Called for each component in turn by
        :func:`validate_component_order`.
        """
        name = self.__class__.__name__
        for key in self.required_inputs:
            if key not in state:
                raise MissingStateInput(
                    name,
                    key,
                    f"{name} reads the model state key '{key}', "
                    "which nothing before it in model.components produces.",
                )

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


def assert_attr_shape(obj, attr, shape):

    assert hasattr(obj, attr), f"{attr} does not exist."
    attr_shape = getattr(obj, attr).shape
    assert (
        attr_shape == shape
    ), f"Expected shape {shape} for {attr} but got {attr_shape}."


@lru_cache(maxsize=1)
def in_tree_components() -> Dict[str, type]:
    """Every concrete component shipped with tabascal, keyed by config reference.

    The keys are the ``"module:Class"`` strings ``model.components`` is written
    in, so they can be quoted straight back at the user. Abstract bases are left
    out -- they cannot be listed in a config.
    """
    found: Dict[str, type] = {}
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{info.name}")
        for name, obj in vars(module).items():
            if (
                isinstance(obj, type)
                and issubclass(obj, Component)
                and obj is not Component
                # Skip re-exports, so each class is named by the module it lives in.
                and obj.__module__ == module.__name__
                and not inspect.isabstract(obj)
            ):
                found[f"{info.name}:{name}"] = obj
    return found


@lru_cache(maxsize=1)
def state_key_producers() -> Dict[str, List[str]]:
    """Map each state key to the in-tree components that write it."""
    producers: Dict[str, List[str]] = {}
    for ref, cls in in_tree_components().items():
        for key in cls.output_shapes:
            producers.setdefault(key, []).append(ref)
    return producers


def component_outputs(component) -> List[str]:
    """The state keys a component writes.

    ``state_outputs`` is what ``Model`` actually merges into the state, so it
    wins once ``setup`` has run. Before that -- or for a component checked
    without an observation to set it up against -- the class-level
    ``output_shapes`` declaration is the same list.
    """
    outputs = getattr(component, "state_outputs", None)
    if not outputs:
        outputs = getattr(component, "output_shapes", None) or {}
    return list(outputs)


def component_ref(component) -> str:
    """The ``"module:Class"`` string a component is written as in a config."""
    cls = component if isinstance(component, type) else type(component)
    for ref, known in in_tree_components().items():
        if known is cls:
            return ref
    return f"{cls.__module__}:{cls.__name__}"


def _describe_missing(
    exc: MissingStateInput, components: Sequence[Any], index: int
) -> str:
    """Turn a missing state key into an actionable message.

    The two failure modes read very differently to a user: a component that is
    not in the list at all has to be added, whereas one that is there but later
    only has to be moved. Say which.
    """
    name = component_ref(components[index])
    later = [
        (i, component_ref(comp))
        for i, comp in enumerate(components[index + 1 :], start=index + 1)
        if exc.key in component_outputs(comp)
    ]
    where = f"model.components[{index}]"

    if later:
        listed = ", ".join(f"'{ref}' (model.components[{i}])" for i, ref in later)
        return (
            f"{where} '{name}' reads the model state key '{exc.key}', which is "
            f"produced by {listed}, listed after it. The components are in the "
            f"wrong order: move the producer before '{name}'."
        )

    producers = state_key_producers().get(exc.key, [])
    if producers:
        listed = ", ".join(f"'{ref}'" for ref in producers)
        absent = (
            "which is not in model.components"
            if len(producers) == 1
            else "none of which are in model.components"
        )
        return (
            f"{where} '{name}' reads the model state key '{exc.key}', which "
            f"nothing before it produces. It is produced by {listed}, {absent} "
            f"-- add one before '{name}'."
        )
    return (
        f"{where} '{name}' reads the model state key '{exc.key}', which nothing "
        "before it produces and which no component in tabascal.components "
        f"writes. Check the spelling of '{exc.key}' in {name}.required_inputs."
    )


def validate_component_order(
    components: Sequence[Any], initial_state_keys: Iterable[str] = ()
) -> None:
    """Check a ``model.components`` list against the components' declarations.

    Walks the list in order, tracking which state keys exist by the time each
    component runs -- ``initial_state_keys`` plus the outputs of everything
    before it -- and raises :class:`ComponentOrderError` on the first component
    whose ``required_inputs`` are not all there. Runs on constructed components
    whether or not they have been set up, so the whole list is checked before any
    of it is traced; without it a missing or mis-ordered component surfaced only
    as a ``KeyError`` on a state key from inside the JIT-ed forward pass.
    """
    state = dict.fromkeys(initial_state_keys)
    for index, comp in enumerate(components):
        try:
            comp.validate_state(state)
        except MissingStateInput as exc:
            raise ComponentOrderError(
                _describe_missing(exc, components, index)
            ) from None
        state.update(dict.fromkeys(component_outputs(comp)))
