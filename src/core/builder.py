from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from tree_sitter import Node as TsNode

from core.parsers.edge import EdgeParser
from core.parsers.node import NodeParser

from .models import DiGraph, Edge, Node


class PyGraphBuilder:
    def __init__(self, root: Path):
        self.root = root
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)

    def build(self) -> DiGraph:
        """
        Build a complete dependency graph from Python source files in the root directory.

        Returns:
            DiGraph containing all nodes and edges found in the codebase
        """
        nodes: Dict[str, Node] = {}
        edges: List[Edge] = []

        # TODO: Ignore gitignore patterns
        for file in self.root.rglob("*.py"):
            try:
                code = file.read_bytes()

                # Skip empty files
                if not code.strip():
                    continue

                tree = self.parser.parse(code)

                # Parse nodes and edges using cursor-based traversal
                file_nodes, file_edges = self._parse_file(
                    code, tree.root_node, str(file)
                )

                # Add nodes to the collection
                for node in file_nodes:
                    nodes[node.id] = node

                # Add edges to the collection
                edges.extend(file_edges)

            except Exception as e:
                print(f"Warning: Failed to parse file {file}: {e}")
                continue

        # Create complete dependency graph
        graph = DiGraph(
            nodes=list(nodes.values()),
            edges=edges,
            bindings={},  # Environment-specific bindings can be added later
        )

        return graph

    def _parse_file(
        self, code: bytes, root_node, filepath: str
    ) -> Tuple[List[Node], List[Edge]]:
        """
        Parse a single file using iterative cursor-based traversal.
        No recursion - uses explicit stack for traversal state management.

        Returns:
            Tuple of (nodes, edges) found in the file
        """
        node_parser = NodeParser(code, filepath)
        edge_parser = EdgeParser(code, filepath)
        nodes = []
        edges = []

        # Add module node first
        module_node = node_parser.create_module_node(root_node)
        nodes.append(module_node)

        # Iterative traversal using explicit stack
        traversal_stack: List[Tuple[TsNode, Optional[str]]] = [(root_node, None)]

        while traversal_stack:
            current_ts_node, parent_id = traversal_stack.pop()
            current_parent = parent_id

            # Let NodeParser process current node
            parsed_node = node_parser.process_node(current_ts_node, parent_id)
            if parsed_node:
                nodes.append(parsed_node)
                current_parent = parsed_node.id

            # Let EdgeParser process current node for relationships
            try:
                parsed_edges = edge_parser.process_node(current_ts_node, current_parent)
                edges.extend(parsed_edges)
            except Exception:
                # Continue parsing even if edge parsing fails for this node
                pass

            # Add children to stack in reverse order to maintain left-to-right processing
            for child in reversed(current_ts_node.children):
                traversal_stack.append((child, current_parent))

        return nodes, edges
