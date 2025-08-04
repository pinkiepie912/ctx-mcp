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

            try:
                # src/parsers/node.py -> src.parsers.node
                module_path = file.relative_to(self.root)
                module_name = str(module_path.with_suffix('')).replace('/', '.')
            except ValueError:
                module_name = file.stem
            
            nodes[module_name] = Node(id=module_name, type=NodeType.MODULE, module=module_name)
            
            root_node = tree.root_node
            cursor = root_node.walk()
            
            reached_root = False
            while not reached_root:
                ts_node = cursor.node
                if not ts_node:
                    continue

                new_nodes = parse_nodes(code, ts_node, module_name)
                for n in new_nodes:
                    nodes[n.id] = n
                
                new_edges = parse_edges(code, ts_node, module_name)
                edges.extend(new_edges)

                if cursor.goto_first_child():
                    continue

                if cursor.goto_next_sibling():
                    continue

                retracking = True
                while retracking:
                    if not cursor.goto_parent():
                        retracking = False
                        reached_root = True

                    if cursor.goto_next_sibling():
                        retracking = False