from pathlib import Path
from typing import List, Optional

from tree_sitter import Node as TsNode

from core.models import Node, NodeParameter, NodeScope, NodeType

from .text import parse_text

__all__ = ["NodeParser"]


class NodeParser:
    """Parser for extracting Node objects from Tree-sitter"""

    # Class-level constants to eliminate duplication
    DEFINITION_TYPES = frozenset(
        {"class_definition", "function_definition", "async_function_definition"}
    )

    DEFINITION_TYPE_MAPPING = {
        "class_definition": NodeType.CLASS,
        "function_definition": NodeType.FUNCTION,
        "async_function_definition": NodeType.FUNCTION,
    }

    def __init__(self, code: bytes, filepath: str):
        self.code = code
        self.filepath = filepath
        self.module_path = self._get_module_path(filepath)

    def create_module_node(self, root_node: TsNode) -> Node:
        """Create module node for the file."""
        return self._create_module_node(root_node)

    def process_node(self, ts_node: TsNode, parent_id: Optional[str]) -> Optional[Node]:
        """
        Process a single node without moving cursor.
        Only analyzes the given node and returns parsed Node if applicable.
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
            id=f"{self.filepath}::module::{self.module_path}",
            type=NodeType.MODULE,
            filepath=self.filepath,
            source_line=(1, root_node.end_point[0] + 1),
            qulified_name=self.module_path,
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
        name = self._extract_name(ts_node, node_type)
        if not name:
            return None

        # Determine scope
        scope = self._determine_scope(node_type, parent_id)

        # Build qualified name
        qualified_name = self._build_qualified_name(name, node_type, parent_id)

        # Create node ID
        node_id = f"{self.filepath}::{node_type.value.lower()}::{name}"
        if parent_id and parent_id != f"{self.filepath}::module::{self.module_path}":
            # For nested nodes, include parent context in ID
            parent_name = parent_id.split("::")[-1]
            node_id = (
                f"{self.filepath}::{node_type.value.lower()}::{parent_name}.{name}"
            )

        # Extract parameters for functions/methods
        parameters = None
        if node_type == NodeType.FUNCTION:
            parameters = self._extract_parameters(ts_node)

        return Node(
            id=node_id,
            type=node_type,
            filepath=self.filepath,
            source_line=(ts_node.start_point[0] + 1, ts_node.end_point[0] + 1),
            qulified_name=qualified_name,
            scope=scope,
            owner_id=parent_id or "",
            parameters=parameters,
        )

    def _extract_name(self, ts_node: TsNode, node_type: NodeType) -> Optional[str]:
        """Extract the name of a class or function from Tree-sitter node"""
        # Use iterative approach instead of recursion
        nodes_to_check = [ts_node]

        while nodes_to_check:
            current_node = nodes_to_check.pop(0)

            # Handle decorated definitions - add the actual definition to check
            if current_node.type == "decorated_definition":
                for child in current_node.children:
                    if child.type in self.DEFINITION_TYPES:
                        nodes_to_check.append(child)
                continue

            # Look for identifier node that contains the name
            for child in current_node.children:
                if child.type == "identifier":
                    return parse_text(self.code, child)

        return None

    def _extract_parameters(self, ts_node: TsNode) -> List[NodeParameter]:
        """Extract function parameters from Tree-sitter node"""
        parameters = []

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
            if param_child.type in [
                "identifier",
                "default_parameter",
                "typed_parameter",
                "typed_default_parameter",
                "list_splat_pattern",
                "dictionary_splat_pattern",
            ]:
                param = self._parse_parameter_node(param_child, position)
                if param:
                    parameters.append(param)
                    position += 1

        return parameters

    def _parse_parameter_node(
        self, param_node: TsNode, position: int
    ) -> Optional[NodeParameter]:
        """Parse a single parameter node"""
        param_name = None
        type_hint = None
        default_value = None
        is_varargs = False
        is_kwargs = False
        required = True

        if param_node.type == "identifier":
            # Simple parameter: def func(param):
            param_name = parse_text(self.code, param_node)

        elif param_node.type == "default_parameter":
            # Parameter with default value: def func(param=default):
            for child in param_node.children:
                if child.type == "identifier":
                    param_name = parse_text(self.code, child)
                elif child.type not in ["=", ","]:  # Skip operators and separators
                    default_value = parse_text(self.code, child)
                    required = False

        elif param_node.type == "typed_parameter":
            # Typed parameter: def func(param: Type):
            for child in param_node.children:
                if child.type == "identifier":
                    param_name = parse_text(self.code, child)
                elif child.type == "type":
                    type_hint = parse_text(self.code, child)

        elif param_node.type == "typed_default_parameter":
            # Typed parameter with default: def func(param: Type = default):
            for child in param_node.children:
                if child.type == "identifier":
                    param_name = parse_text(self.code, child)
                elif child.type == "type":
                    type_hint = parse_text(self.code, child)
                elif child.type not in [":", "=", ","]:  # Skip operators
                    if "=" in parse_text(self.code, param_node):
                        default_value = parse_text(self.code, child)
                        required = False

        elif param_node.type == "list_splat_pattern":
            # *args parameter
            for child in param_node.children:
                if child.type == "identifier":
                    param_name = parse_text(self.code, child)
            is_varargs = True
            required = False

        elif param_node.type == "dictionary_splat_pattern":
            # **kwargs parameter
            for child in param_node.children:
                if child.type == "identifier":
                    param_name = parse_text(self.code, child)
            is_kwargs = True
            required = False

        if param_name:
            return NodeParameter(
                name=param_name,
                type_hint=type_hint,
                default_value=default_value,
                is_varargs=is_varargs,
                is_kwargs=is_kwargs,
                position=position,
                required=required,
            )

        return None

    def _determine_scope(
        self, node_type: NodeType, parent_id: Optional[str]
    ) -> NodeScope:
        """Determine the scope of a node based on its context"""
        if not parent_id:
            return NodeScope.MODULE

        # Check parent type from parent_id
        if "::class::" in parent_id:
            return (
                NodeScope.METHOD if node_type == NodeType.FUNCTION else NodeScope.CLASS
            )
        elif "::function::" in parent_id:
            return NodeScope.LOCAL
        else:
            return NodeScope.MODULE

    def _build_qualified_name(
        self, name: str, node_type: NodeType, parent_id: Optional[str]
    ) -> str:
        """Build the fully qualified name for the node"""
        qualified_parts = [self.module_path]

        if parent_id and parent_id != f"{self.filepath}::module::{self.module_path}":
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
        """Extract qualified name from a node ID"""
        # Node ID format: filepath::type::name
        parts = node_id.split("::")
        if len(parts) >= 3:
            name_part = parts[2]
            # Handle nested names like "ParentClass.method_name"
            if "." in name_part:
                return f"{self.module_path}.{name_part}"
            else:
                return f"{self.module_path}.{name_part}"
        return self.module_path
