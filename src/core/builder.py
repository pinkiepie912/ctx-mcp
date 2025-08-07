from pathlib import Path
from typing import Dict, List

from tree_sitter import Language, Parser, TreeCursor
import tree_sitter_python as tspython

from .models import Edge, Node
from core.parsers.node import NodeParser

import pprint


class PyGraphBuilder:
    def __init__(self, root: Path):
        self.root = root
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)

    def build(self) -> None:
        """
        Build a graph from the Python source files in the root directory.
        """
        nodes: Dict[str, Node] = {}
        edges: List[Edge] = []

        # TODO: Ignore gitignore patterns
        for file in self.root.rglob("*.py"):
            if not file.name == "user_reader.py":
                continue
            code = file.read_bytes()
            tree = self.parser.parse(code)

            # Parse nodes using cursor-based traversal
            file_nodes = self._parse_file_with_cursor(code, tree.root_node, str(file))

            # Add nodes to the collection
            for node in file_nodes:
                nodes[node.id] = node

            # TODO: Parse edges for this file using the same cursor traversal
            # file_edges = parse_edges(code, root_node, str(file))
            # edges.extend(file_edges)

        for row in nodes.values():
            pprint.pprint(row.dict())
            print("\n")

    def _parse_file_with_cursor(
        self, code: bytes, root_node, filepath: str
    ) -> List[Node]:
        """
        Parse a single file using iterative cursor-based traversal.
        No recursion - uses explicit stack for traversal state management.
        """
        node_parser = NodeParser(code, filepath)
        nodes = []

        # Add module node first
        module_node = node_parser.create_module_node(root_node)
        nodes.append(module_node)

        # Iterative traversal using explicit stack - no recursion
        traversal_stack = [(root_node, None)]  # (ts_node, parent_id) tuples

        while traversal_stack:
            current_ts_node, parent_id = traversal_stack.pop()
            current_parent = parent_id

            # Let NodeParser process current node
            parsed_node = node_parser.process_node(current_ts_node, parent_id)
            if parsed_node:
                nodes.append(parsed_node)
                current_parent = parsed_node.id

            # TODO: EdgeParser will also process current_ts_node here
            # parsed_edges = edge_parser.process_node(current_ts_node, current_parent)
            # edges.extend(parsed_edges)

            # Add children to stack in reverse order to maintain left-to-right processing
            for child in reversed(current_ts_node.children):
                traversal_stack.append((child, current_parent))

        return nodes
