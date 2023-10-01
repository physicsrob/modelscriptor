from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.graph import Node, Concatenate, Linear
from modelscriptor.compiler.components.component import NodeComponentStrategy, Component


class AttnLayerComponent(Component):
    ...
