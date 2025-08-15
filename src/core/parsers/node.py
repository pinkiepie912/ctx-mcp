from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tree_sitter import Node as TsNode

from core.models import Node, NodeParameter, NodeScope, NodeType

from .text import parse_text


@dataclass(slots=True, frozen=True)
class ParameterData:
    """
    Data structure for parsed parameter information.

    This immutable structure contains all information extracted from
    a Tree-sitter parameter node, providing type-safe access to
    parameter attributes.

    Attributes:
        name: Parameter name (e.g., 'param', 'args', 'kwargs')
        type_hint: Type annotation if present (e.g., 'str', 'List[int]')
        default_value: Default value expression if present (e.g., 'None', '[]')
        is_varargs: True for *args parameters
        is_kwargs: True for **kwargs parameters
        required: False if parameter has default value or is *args/**kwargs
    """

    name: Optional[str] = None
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_varargs: bool = False
    is_kwargs: bool = False
    required: bool = True


__all__ = ["NodeParser", "ParameterData"]


class NodeParser:
    """
    Parser for extracting Node objects from Tree-sitter AST.

    This class provides efficient parsing of Python source code to extract
    structural information including modules, classes, functions, and their
    parameters. Uses iterative algorithms and robust error handling for
    scalability and reliability.

    Attributes:
        code: Raw bytes of the source code
        filepath: Path to the source file
        module_path: Python module path derived from filepath

    Constants:
        ID_SEPARATOR: Separator used in node IDs
        OPERATORS_TO_SKIP: Operators to ignore during parameter parsing
        DEFINITION_TYPES: Tree-sitter node types that represent definitions
        DEFINITION_TYPE_MAPPING: Maps Tree-sitter types to NodeType enums
        PARAMETER_TYPES: Tree-sitter node types for function parameters
    """

    # Class-level constants to eliminate duplication
    ID_SEPARATOR = "::"
    OPERATORS_TO_SKIP = {"=", ":", ","}

    DEFINITION_TYPES = frozenset(
        {"class_definition", "function_definition", "async_function_definition"}
    )

    DEFINITION_TYPE_MAPPING = {
        "class_definition": NodeType.CLASS,
        "function_definition": NodeType.FUNCTION,
        "async_function_definition": NodeType.ASYNC_FUNCTION,
    }

    PARAMETER_TYPES = frozenset(
        {
            "identifier",
            "default_parameter",
            "typed_parameter",
            "typed_default_parameter",
            "list_splat_pattern",
            "dictionary_splat_pattern",
        }
    )

    def __init__(self, code: bytes, filepath: str):
        """
        Initialize NodeParser with source code and file path.

        Args:
            code: Raw bytes of Python source code
            filepath: Path to the source file

        Raises:
            ValueError: If code is empty or filepath is invalid
        """
        if not code:
            raise ValueError("Code cannot be empty")
        if not filepath or not filepath.strip():
            raise ValueError("Filepath cannot be empty")

        self.code = code
        self.filepath = filepath
        self.module_path = self._get_module_path(filepath)

    def create_module_node(self, root_node: TsNode) -> Node:
        """Create module node for the file."""
        return self._create_module_node(root_node)

    def process_node(self, ts_node: TsNode, parent_id: Optional[str]) -> Optional[Node]:
        """
        Process a single Tree-sitter node and extract structural information.

        This method analyzes a Tree-sitter node to determine if it represents
        a Python definition (class, function, etc.) and creates a corresponding
        Node object with complete metadata.

        Args:
            ts_node: Tree-sitter node to process
            parent_id: ID of the parent node for hierarchy tracking

        Returns:
            Node object if the Tree-sitter node represents a definition,
            None otherwise
        """
        node_type = self._get_node_type(ts_node)

        if node_type:
            return self._create_node(ts_node, node_type, parent_id)

        return None

    def _get_module_path(self, filepath: str) -> str:
        """Convert file path to Python module path"""
        path = Path(filepath)
        # Remove .py extension and convert path separators to dots
        module_parts = path.with_suffix("").parts

        # Remove common prefixes like 'src', handle relative paths
        if "src" in module_parts:
            src_index = module_parts.index("src")
            module_parts = module_parts[src_index + 1 :]

        return ".".join(module_parts)

    def _create_module_node(self, root_node: TsNode) -> Node:
        """Create a MODULE type node for the file itself"""
        return Node(
            id=f"{self.filepath}{self.ID_SEPARATOR}module{self.ID_SEPARATOR}{self.module_path}",
            type=NodeType.MODULE,
            filepath=self.filepath,
            source_line=(1, root_node.end_point[0] + 1),
            qualified_name=self.module_path,
            scope=NodeScope.MODULE,
            owner_id="",
        )

    def _get_node_type(self, ts_node: TsNode) -> Optional[NodeType]:
        """Map Tree-sitter node types to NodeType enum"""
        # Handle decorated definitions
        if ts_node.type == "decorated_definition":
            # Look for the actual definition in children
            for child in ts_node.children:
                if child.type in self.DEFINITION_TYPE_MAPPING:
                    return self.DEFINITION_TYPE_MAPPING[child.type]

        return self.DEFINITION_TYPE_MAPPING.get(ts_node.type)

    def _create_node(
        self, ts_node: TsNode, node_type: NodeType, parent_id: Optional[str]
    ) -> Optional[Node]:
        """Create a Node object from Tree-sitter node"""
        name = self._extract_name(ts_node)
        if not name:
            return None

        # Determine scope
        scope = self._determine_scope(node_type, parent_id)

        # Build qualified name
        qualified_name = self._build_qualified_name(name, parent_id)

        # Create node ID
        node_id = (
            f"{self.filepath}{self.ID_SEPARATOR}"
            f"{node_type.value.lower()}{self.ID_SEPARATOR}{name}"
        )
        module_node_id = f"{self.filepath}{self.ID_SEPARATOR}module{self.ID_SEPARATOR}{self.module_path}"
        if parent_id and parent_id != module_node_id:
            # For nested nodes, include parent context in ID
            parent_name = parent_id.split(self.ID_SEPARATOR)[-1]
            node_id = (
                f"{self.filepath}{self.ID_SEPARATOR}{node_type.value.lower()}"
                f"{self.ID_SEPARATOR}{parent_name}.{name}"
            )

        # Extract parameters for functions/methods (both sync and async)
        parameters = None
        if node_type in (NodeType.FUNCTION, NodeType.ASYNC_FUNCTION):
            parameters = self._extract_parameters(ts_node)

        return Node(
            id=node_id,
            type=node_type,
            filepath=self.filepath,
            source_line=(ts_node.start_point[0] + 1, ts_node.end_point[0] + 1),
            qualified_name=qualified_name,
            scope=scope,
            owner_id=parent_id or "",
            parameters=parameters,
        )

    def _extract_name(self, ts_node: TsNode) -> Optional[str]:
        """Extract the name of a class or function from Tree-sitter node"""
        nodes_to_check = deque([ts_node])

        while nodes_to_check:
            current_node = nodes_to_check.popleft()

            # Handle decorated definitions - add the actual definition to check
            if current_node.type == "decorated_definition":
                for child in current_node.children:
                    if child.type in self.DEFINITION_TYPES:
                        nodes_to_check.append(child)
                continue

            # Look for identifier node that contains the name
            for child in current_node.children:
                if child.type == "identifier":
                    try:
                        return parse_text(self.code, child)
                    except Exception:
                        # Skip malformed nodes
                        continue

        return None

    def _extract_parameters(self, ts_node: TsNode) -> List[NodeParameter]:
        """Extract function parameters from Tree-sitter node"""
        parameters = []

        try:
            # Find the parameters node in the function definition
            params_node = None
            for child in ts_node.children:
                if child.type == "parameters":
                    params_node = child
                    break

            if not params_node:
                return parameters

            position = 0
            for param_child in params_node.children:
                if param_child.type in self.PARAMETER_TYPES:
                    try:
                        param = self._parse_parameter_node(param_child, position)
                        if param:
                            parameters.append(param)
                            position += 1
                    except Exception:
                        # Skip malformed parameter nodes
                        continue

        except Exception:
            # Return partial results if parsing fails
            pass

        return parameters

    def _parse_parameter_node(
        self, param_node: TsNode, position: int
    ) -> Optional[NodeParameter]:
        """Parse a single parameter node by delegating to specific parsers"""
        parser_map = {
            "identifier": self._parse_simple_parameter,
            "default_parameter": self._parse_default_parameter,
            "typed_parameter": self._parse_typed_parameter,
            "typed_default_parameter": self._parse_typed_default_parameter,
            "list_splat_pattern": self._parse_varargs_parameter,
            "dictionary_splat_pattern": self._parse_kwargs_parameter,
        }

        parser = parser_map.get(param_node.type)
        if parser:
            param_data = parser(param_node)
            if param_data and param_data.name:
                return NodeParameter(
                    name=param_data.name,
                    type_hint=param_data.type_hint,
                    default_value=param_data.default_value,
                    is_varargs=param_data.is_varargs,
                    is_kwargs=param_data.is_kwargs,
                    position=position,
                    required=param_data.required,
                )

        return None

    def _parse_simple_parameter(self, param_node: TsNode) -> Optional[ParameterData]:
        """Parse simple parameter: def func(param):"""
        try:
            name = parse_text(self.code, param_node)
            return ParameterData(name=name) if name else None
        except Exception:
            return None

    def _parse_default_parameter(self, param_node: TsNode) -> Optional[ParameterData]:
        """Parse parameter with default: def func(param=default):"""
        param_name = None
        default_value = None

        try:
            for child in param_node.children:
                if child.type == "identifier":
                    param_name = parse_text(self.code, child)
                elif child.type not in self.OPERATORS_TO_SKIP:
                    default_value = parse_text(self.code, child)
        except Exception:
            return None

        return (
            ParameterData(name=param_name, default_value=default_value, required=False)
            if param_name
            else None
        )

    def _parse_typed_parameter(self, param_node: TsNode) -> Optional[ParameterData]:
        """Parse typed parameter: def func(param: Type):"""
        param_name = None
        type_hint = None

        for child in param_node.children:
            if child.type == "identifier":
                param_name = parse_text(self.code, child)
            elif child.type == "type":
                type_hint = parse_text(self.code, child)

        return (
            ParameterData(name=param_name, type_hint=type_hint) if param_name else None
        )

    def _parse_typed_default_parameter(
        self, param_node: TsNode
    ) -> Optional[ParameterData]:
        """Parse typed parameter with default: def func(param: Type = default):"""
        param_name = None
        type_hint = None
        default_value = None
        required = True

        for child in param_node.children:
            if child.type == "identifier":
                param_name = parse_text(self.code, child)
            elif child.type == "type":
                type_hint = parse_text(self.code, child)
            elif child.type not in self.OPERATORS_TO_SKIP:
                if "=" in parse_text(self.code, param_node):
                    default_value = parse_text(self.code, child)
                    required = False

        return (
            ParameterData(
                name=param_name,
                type_hint=type_hint,
                default_value=default_value,
                required=required,
            )
            if param_name
            else None
        )

    def _parse_varargs_parameter(self, param_node: TsNode) -> Optional[ParameterData]:
        """Parse *args parameter"""
        param_name = None
        for child in param_node.children:
            if child.type == "identifier":
                param_name = parse_text(self.code, child)
                break

        return (
            ParameterData(name=param_name, is_varargs=True, required=False)
            if param_name
            else None
        )

    def _parse_kwargs_parameter(self, param_node: TsNode) -> Optional[ParameterData]:
        """Parse **kwargs parameter"""
        param_name = None
        for child in param_node.children:
            if child.type == "identifier":
                param_name = parse_text(self.code, child)
                break

        return (
            ParameterData(name=param_name, is_kwargs=True, required=False)
            if param_name
            else None
        )

    def _determine_scope(
        self, node_type: NodeType, parent_id: Optional[str]
    ) -> NodeScope:
        """Determine the scope of a node based on its context"""
        if not parent_id:
            return NodeScope.MODULE

        # Check parent type from parent_id
        if f"{self.ID_SEPARATOR}class{self.ID_SEPARATOR}" in parent_id:
            return (
                NodeScope.METHOD
                if node_type in (NodeType.FUNCTION, NodeType.ASYNC_FUNCTION)
                else NodeScope.CLASS
            )
        elif f"{self.ID_SEPARATOR}function{self.ID_SEPARATOR}" in parent_id:
            return NodeScope.LOCAL
        else:
            return NodeScope.MODULE

    def _build_qualified_name(self, name: str, parent_id: Optional[str]) -> str:
        """Build the fully qualified name for the node"""
        qualified_parts = [self.module_path]

        module_node_id = f"{self.filepath}{self.ID_SEPARATOR}module{self.ID_SEPARATOR}{self.module_path}"
        if parent_id and parent_id != module_node_id:
            # Extract parent name from parent_id
            parent_qualified = self._extract_qualified_from_id(parent_id)
            if parent_qualified and parent_qualified != self.module_path:
                # Remove module path prefix if it exists
                if parent_qualified.startswith(self.module_path + "."):
                    parent_qualified = parent_qualified[len(self.module_path) + 1 :]
                qualified_parts.append(parent_qualified)

        qualified_parts.append(name)
        return ".".join(qualified_parts)

    def _extract_qualified_from_id(self, node_id: str) -> str:
        """
        Extract qualified name from a node ID.

        Args:
            node_id: Node ID in format "filepath::type::name"

        Returns:
            Fully qualified name (e.g. "module.ClassName" or "module.ParentClass.method_name")
        """
        # Node ID format: filepath::type::name
        parts = node_id.split(self.ID_SEPARATOR)
        if len(parts) >= 3:
            name_part = parts[2]
            # Both nested and top-level names get module path prefix
            return f"{self.module_path}.{name_part}"
        return self.module_path
