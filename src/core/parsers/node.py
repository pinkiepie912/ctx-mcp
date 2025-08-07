from typing import List, Optional, Dict, Set
from tree_sitter import Node as TsNode
from pathlib import Path

from .text import parse_text

from core.models import Node, NodeType, NodeScope

__all__ = ["parse_nodes"]


def parse_nodes(code: bytes, ts_node: TsNode, filepath: str) -> List[Node]:
    """
    Parse Python AST nodes and convert them to Node objects.

    Args:
        code: Source code bytes
        ts_node: Tree-sitter root node
        filepath: File path for the source

    Returns:
        List of parsed Node objects
    """
    parser = _NodeParser(code, filepath)
    return parser.parse(ts_node)


class _NodeParser:
    """Parser for extracting Node objects from Tree-sitter"""

    def __init__(self, code: bytes, filepath: str):
        self.code = code
        self.filepath = filepath
        self.module_path = self._get_module_path(filepath)
        self.nodes: List[Node] = []
        self.scope_stack: List[str] = []  # Track nested scopes

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

    def parse(self, root_node: TsNode) -> List[Node]:
        """Parse the AST starting from root node"""
        # Add module node
        module_node = self._create_module_node(root_node)
        self.nodes.append(module_node)

        # Parse child nodes recursively
        self._parse_node_recursive(root_node, None)

        return self.nodes

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

    def _parse_node_recursive(self, ts_node: TsNode, parent_id: Optional[str]):
        """Recursively parse Tree-sitter nodes"""
        node_type = self._get_node_type(ts_node)

        if node_type:
            node = self._create_node(ts_node, node_type, parent_id)
            if node:
                self.nodes.append(node)
                current_parent = node.id
            else:
                current_parent = parent_id
        else:
            current_parent = parent_id

        # Parse children
        for child in ts_node.children:
            self._parse_node_recursive(child, current_parent)

    def _get_node_type(self, ts_node: TsNode) -> Optional[NodeType]:
        """Map Tree-sitter node types to NodeType enum"""
        type_mapping = {
            "class_definition": NodeType.CLASS,
            "function_definition": NodeType.FUNCTION,
            "async_function_definition": NodeType.FUNCTION,
        }

        # Handle decorated definitions
        if ts_node.type == "decorated_definition":
            # Look for the actual definition in children
            for child in ts_node.children:
                if child.type in type_mapping:
                    return type_mapping[child.type]

        return type_mapping.get(ts_node.type)

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

        return Node(
            id=node_id,
            type=node_type,
            filepath=self.filepath,
            source_line=(ts_node.start_point[0] + 1, ts_node.end_point[0] + 1),
            qulified_name=qualified_name,
            scope=scope,
            owner_id=parent_id or "",
        )

    def _extract_name(self, ts_node: TsNode, node_type: NodeType) -> Optional[str]:
        """Extract the name of a class or function from Tree-sitter node"""
        # Handle decorated definitions
        if ts_node.type == "decorated_definition":
            for child in ts_node.children:
                if child.type in [
                    "class_definition",
                    "function_definition",
                    "async_function_definition",
                ]:
                    return self._extract_name(child, node_type)

        # Look for identifier node that contains the name
        for child in ts_node.children:
            if child.type == "identifier":
                return parse_text(self.code, child)

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
