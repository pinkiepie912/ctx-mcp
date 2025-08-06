from typing import List
from tree_sitter import Node as TsNode
from models import Edge, EdgeKind

__all__ = ["parse_edges"]


def _parse_text(code: bytes, node: TsNode) -> str:
    return code[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def parse_edges(code: bytes, ts_node: TsNode, module_name: str) -> List[Edge]:
    """
    Parses a tree-sitter node to extract edges.
    Currently supports import statements.
    """
    edges = []
    if ts_node.type == "import_statement":
        # e.g. import a.b, c.d
        name_nodes = ts_node.children_by_field_name("name")
        for name_node in name_nodes:
            target = _parse_text(code, name_node)
            edges.append(Edge(kind=EdgeKind.IMPORTS, source=module_name, target=target))

    elif ts_node.type == "import_from_statement":
        # e.g. from a.b import c, d
        # The edge is from the current module to the module being imported from.
        module_name_node = ts_node.child_by_field_name("module_name")
        if module_name_node:
            target = _parse_text(code, module_name_node)
            # TODO: Handle relative imports.
            edges.append(Edge(kind=EdgeKind.IMPORTS, source=module_name, target=target))

    return edges
