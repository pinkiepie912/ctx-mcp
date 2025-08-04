from enum import Enum
from typing import List, Dict, Optional

from pydantic import BaseModel


class NodeType(Enum):
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    TOKEN = "TOKEN"
    MODULE = "MODULE"

class ScopeType(Enum):
    APP = "APP"
    REQUEST = "REQUEST"
    TASK = "TASK"
    SINGLETON = "SINGLETON"


class EdgeKind(Enum):
    INJECTS = "INJECTS"  # A -> token/class
    PROVIDES = "PROVIDES"  # provider -> token
    IMPLEMENTS = "IMPLEMENTS"  # class -> interface
    IMPORTS = "IMPORTS"  # module -> module
    CALLS = "CALLS"  # impl|endpoint -> external service 
    PUBLISH = "PUBLISHES"  # producer -> event topic
    SBSCRIBES = "SUBSCRIBES"  # consumer -> event topic 


class Node(BaseModel):
    id: str
    type: NodeType
    module: Optional[str] = None
    scope: Optional[ScopeType] = None
    implements: List[str] = []
    capabilities: List[str] = []


class Edge(BaseModel):
    # injects: A -> token/class, provides: provider -> token
    kind: EdgeKind
    source: str
    target: str
    or_group: Optional[str] = None
    optional: bool = False


class DiGraph(BaseModel):
    version: str = "0.1.0"
    env: str = "dev"
    nodes: List[Node] = []
    edges: List[Edge] = []
    bindings: Dict[str, Dict[str, str]] = {}  # env -> token->impl
