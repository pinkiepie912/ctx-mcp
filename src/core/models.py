from enum import Enum
from typing import List, Dict, Optional, Tuple

from pydantic import BaseModel


class NodeType(Enum):
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    TOKEN = "TOKEN"
    MODULE = "MODULE"


class NodeScope(Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    LOCAL = "local"


class NodeParameter(BaseModel):
    """
    Parameters for a node, used for dependency injection

    Attributes:
        name: Name of the parameter (e.g. 'order_repo')
        type_hint: Optional type hint for the parameter (e.g. 'OrderRepository')
        default_value: Optional default value for the parameter (e.g. 'None' or 'default_repo')
        is_varargs: Whether this parameter accepts variable arguments (e.g. *args)
        is_kwargs: Whether this parameter accepts keyword arguments (e.g. **kwargs)
        position: Position of the parameter in the function signature (0-based index)
        required: Whether this parameter is required (default=True)
    """

    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_varargs: bool = False
    is_kwargs: bool = False
    position: int
    required: bool = True


class Node(BaseModel):
    """
    Node model

    Attributes:
        id: Unique identifier for the node (e.g. app/services/order_service.py::class::OrderService )
        type: Type of the node (e.g. CLASS, FUNCTION, TOKEN, MODULE)
        filepath: Path to the source file where the node is defined (e.g. src/services/order_service.py)
        source_line: Tuple containing the start and end line numbers of the node definition in the source file
        qulified_name: Fully qualified name of the node (e.g. app.services.order_service.OrderService )
        scope: Scope of the node declaration (e.g. module, class, function, method, local)
        owner_id: ID of the owner node (e.g. class for method, module for function)
    """

    id: str
    type: NodeType
    filepath: str
    source_line: Tuple[int, int]  # (line_start, line_end)
    qulified_name: str
    scope: NodeScope
    owner_id: str
    parameters: Optional[List[NodeParameter]] = None


class EdgeKind(Enum):
    INJECTS = "INJECTS"  # A -> token/class
    PROVIDES = "PROVIDES"  # provider -> token
    IMPLEMENTS = "IMPLEMENTS"  # class -> interface
    IMPORTS = "IMPORTS"  # module -> module
    CALLS = "CALLS"  # impl|endpoint -> external service
    PUBLISHES = "PUBLISHES"  # producer -> event topic
    SUBSCRIBES = "SUBSCRIBES"  # consumer -> event topic


class InjectionType(Enum):
    CONSTRUCTOR = "CONSTRUCTOR"  # Constructor injection
    SETTER = "SETTER"  # Setter injection
    INTERFACE = "INTERFACE"  # Interface injection
    FIELD = "FIELD"  # Field injection
    FACTORY = "FACTORY"  # Injection via factory
    CONTAINER = "CONTAINER"  # Container-based injection


class AccessType(Enum):
    DIRECT = "DIRECT"  # Direct access
    ATTRIBUTE = "ATTRIBUTE"  # Attribute access (obj.attr)
    METHOD = "METHOD"  # Method call (obj.method())
    IMPORT = "IMPORT"  # Access via import
    INHERITANCE = "INHERITANCE"  # Access via inheritance
    ASYNC = "ASYNC"  # Asynchronous call
    EVENT = "EVENT"  # Access via event


class Edge(BaseModel):
    """
    Edge model representing relationships between code elements

    Attributes:
        kind: Type of relationship between nodes (INJECTS, PROVIDES, IMPLEMENTS, IMPORTS, CALLS, PUBLISHES, SUBSCRIBES)
        source: Unique identifier of the source node in the relationship (e.g. app/services/order_service.py::class::OrderService)
        target: Unique identifier of the target node in the relationship (e.g. app/repositories/order_repo.py::class::OrderRepository)
        source_location: Location where the relationship occurs in source code (filepath:line_number format, e.g. "app/services/order_service.py:25")
        weight: Strength/importance of the relationship from 1-10 (1=low, 10=critical, default=1)
        access_type: How the target is accessed from source (direct, attribute, method, import, inheritance, async, event)

        injection_type: Type of dependency injection for INJECTS edges (constructor, setter, interface, field, factory, container)
        or_group: Logical grouping for alternative dependencies (e.g. 'cache_provider' for Redis OR InMemory cache)
        optional: Whether this dependency is optional and the source can work without it (default=False)
        async_context: Whether the relationship occurs in async/await context (default=False)
        environment_specific: Environment where this relationship applies (dev, test, prod, staging, or None for all)
    """

    kind: EdgeKind
    source: str
    target: str
    source_line: Tuple[int, int]  # (line_start, line_end)
    weight: int = 1
    access_type: AccessType = AccessType.DIRECT
    injection_type: Optional[InjectionType] = None
    or_group: Optional[str] = None
    optional: bool = False
    async_context: bool = False
    test_only: bool = False


class DiGraph(BaseModel):
    version: str = "0.1.0"
    env: str = "dev"
    nodes: List[Node] = []
    edges: List[Edge] = []
    bindings: Dict[str, Dict[str, str]] = {}  # env -> token->impl
