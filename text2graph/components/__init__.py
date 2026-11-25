"""Pipeline components, grouped by what they do, plus the `Component` base class
and the registry that lets configs refer to components by name.

Every component class is importable from here, e.g.
`from text2graph.components import GCNPropagator`.
"""

from .component import Component, RepeatComponent
from .component_registry import register_component, get_component

from . import node_creation, edge_creation, feature_extraction, labeling, propagating, sampling

from .node_creation import *
from .edge_creation import *
from .feature_extraction import *
from .labeling import *
from .propagating import *
from .sampling import *

__all__ = [
    'Component',
    'RepeatComponent',
    'register_component',
    'get_component',
    *node_creation.__all__,
    *edge_creation.__all__,
    *feature_extraction.__all__,
    *labeling.__all__,
    *propagating.__all__,
    *sampling.__all__,
]
