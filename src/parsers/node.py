from typing import List
from tree_sitter import Node as TsNode

from src.parsers.test import test_fuck

from ..models import Node, NodeType

__all__ = ["parse_nodes"]


def parse_nodes(code: bytes, ts_node: TsNode, module_name: str) -> List[Node]:
    """
        Parses a tree-sitter node to extract graph nodes.
        Currently supports class and function definitions.
    """
    nodes = []
    if ts_node.type == "class_definition":
        name_node = ts_node.child_by_field_name("name")
        if name_node:
            class_name = parse_text(code, name_node)
            node_id = f"{module_name}.{class_name}"
            nodes.append(Node(id=node_id, type=NodeType.CLASS, module=module_name))
            
    elif ts_node.type == "function_definition":
        name_node = ts_node.child_by_field_name("name")
        if name_node:
            function_name = parse_text(code, name_node)
            node_id = f"{module_name}.{function_name}"
            nodes.append(Node(id=node_id, type=NodeType.FUNCTION, module=module_name))

    return nodes
