from pathlib import Path
from typing import Dict, List, Optional

from ruamel.yaml import YAML
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

from models import DiGraph, Edge, Node, NodeType
from parsers.node import parse_nodes
from parsers.edge import parse_edges

import pprint


class PyGraphBuilder:
    def __init__(self, root: Path, arch_yaml: Optional[Path] = None):
        self.root = root
        # self.arch_cfg = YAML(typ="safe").load(arch_yaml.read_text())
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
            code = file.read_bytes()
            tree = self.parser.parse(code)

            # Parse nodes using the node parser
            root_node = tree.root_node
            file_nodes = parse_nodes(code, root_node, str(file))
            
            # Add nodes to the collection
            for node in file_nodes:
                nodes[node.id] = node
                
            # # Parse edges for this file  
            # file_edges = parse_edges(code, root_node, str(file))
            # edges.extend(file_edges)

        for row in nodes.values():
            pprint.pprint(row.dict())
            print("\n")
