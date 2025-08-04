from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from .models import DiGraph, Node, Edge, Policies
from ruamel.yaml import YAML


def layer_of(filepath: str, layer_globs: Dict[str, List[str]]) -> Optional[str]:
    from fnmatch import fnmatch

    p = filepath.replace("\\", "/")
    for layer, patterns in layer_globs.items():
        if any(fnmatch(p, pat) for pat in patterns):
            return layer
    return None


class PyGraphBuilder:
    def __init__(self, root: Path, arch_yaml: Path):
        self.root = Path(root)
        self.arch_cfg = YAML(typ="safe").load(arch_yaml.read_text())
        self.language = get_language("python")
        self.parser = Parser(self.language)

    def _parse(self, code: bytes):
        return self.parser.parse(code)

    def build(self) -> DiGraph:
        nodes: Dict[str, Node] = {}
        edges: List[Edge] = []

        layer_globs: Dict[str, List[str]] = self.arch_cfg.get("layers", {})
        policies = Policies(
            forbid_new=self.arch_cfg.get("rules", {}).get("forbid_new", []),
            allowed_imports=self.arch_cfg.get("rules", {}).get("allowed_imports", {}),
        )

        # bindings
        bindings = self.arch_cfg.get("bindings", {})
        env = self.arch_cfg.get("env", "dev")

        # project walk
        for file in self.root.rglob("*.py"):
            code = file.read_bytes()
            tree = self._parse(code)
            module = self._module_name(file)
            layer = layer_of(str(file), layer_globs) or "unknown"

            # 1) imports -> edges(kind=imports)
            for src, tgt in self._iter_imports(code, tree):
                edges.append(Edge(kind="imports", source=src, target=tgt))

            # 2) classes/functions
            for cls_name, scope in self._iter_classes(code, tree):
                nid = cls_name
                nodes[nid] = Node(id=nid, type="class", module=module, scope=scope)

            for fn_name in self._iter_functions(code, tree):
                nid = fn_name
                nodes.setdefault(nid, Node(id=nid, type="function", module=module))

            # 3) injection edges
            #    - dependency_injector: default=Provide[Container.Service]
            #    - fastapi.Depends(...)
            for src, tgt, or_group, optional in self._iter_injections(code, tree):
                edges.append(
                    Edge(
                        kind="injects",
                        source=src,
                        target=tgt,
                        or_group=or_group,
                        optional=optional,
                    )
                )

            # 4) provides edges (Container providers → token)
            for provider, token in self._iter_providers(code, tree):
                nodes.setdefault(token, Node(id=token, type="token"))
                edges.append(Edge(kind="provides", source=provider, target=token))

        return DiGraph(
            env=env,
            nodes=list(nodes.values()),
            edges=edges,
            policies=policies,
            bindings=bindings,
        )

    # ======== Tree-sitter utilities (heuristic-level) ========
    def _module_name(self, path: Path) -> str:
        rel = path.relative_to(self.root).with_suffix("")
        return ".".join(rel.parts)

    def _iter_imports(self, code: bytes, tree) -> List[Tuple[str, str]]:
        # returns (source_module, target_module)
        # lightweight: read import/from nodes and map to module names
        results = []
        cursor = tree.walk()

        def walk(n):
            if n.type in ("import_statement", "import_from_statement"):
                src = "<module>"
                # map to imported module string
                tgt = self._text(code, n).strip()
                results.append((src, tgt))
            for i in range(n.child_count):
                walk(n.child(i))

        walk(tree.root_node)
        return results

    def _iter_classes(self, code: bytes, tree) -> List[Tuple[str, Optional[str]]]:
        # returns (qualified_class_name, scope)
        out = []
        cursor = tree.walk()

        def walk(n, parents: List[str]):
            if n.type == "class_definition":
                name_node = n.child_by_field_name("name")
                name = self._text(code, name_node)
                qn = ".".join(parents + [name]) if parents else name
                scope = None  # 추후 데코레이터/메타데이터로 스코프 힌트 가능
                out.append((qn, scope))
                # dive into class body to find nested
                suite = n.child_by_field_name("body")
                if suite:
                    for i in range(suite.child_count):
                        walk(suite.child(i), parents + [name])
            else:
                for i in range(n.child_count):
                    walk(n.child(i), parents)

        walk(tree.root_node, [])
        return out

    def _iter_functions(self, code: bytes, tree) -> List[str]:
        res = []

        def walk(n, parents):
            if n.type == "function_definition":
                name_node = n.child_by_field_name("name")
                name = self._text(code, name_node)
                qn = ".".join(parents + [name]) if parents else name
                res.append(qn)
            for i in range(n.child_count):
                walk(n.child(i), parents)

        walk(tree.root_node, [])
        return res

    def _iter_injections(self, code: bytes, tree):
        """
        Heuristics:
        - @inject decorator + param default Provide[Container.Service]
        - param default = fastapi.Depends(something)
        Returns tuples (source(owner), target(token_or_class), or_group, optional)
        """
        results = []
        text = code.decode("utf-8", errors="ignore")

        # 1) dependency_injector: Provide[Container.X]
        # crude regex fallback for MVP (tree-sitter nodes are verbose for calls/subscripts)
        import re

        provide_pattern = re.compile(r"Provide\[(?P<token>[\w\.]+)\]")
        # find function/class __init__ with defaults containing Provide[...]
        # simple pass: map any Provide[...] within file as injection edge from "<module>"
        for m in provide_pattern.finditer(text):
            token = m.group("token")  # e.g., Container.payment_gateway
            results.append(("<module>", token, None, False))

        # 2) fastapi.Depends(Foo) → inject Foo
        depends_pattern = re.compile(r"Depends\((?P<dep>[\w\.]+)")
        for m in depends_pattern.finditer(text):
            dep = m.group("dep")
            results.append(("<endpoint>", dep, None, True))

        return results

    def _iter_providers(self, code: bytes, tree):
        """
        Very light: detect patterns like:
          container = Container()
          container.payment_gateway = providers.Factory(StripeGateway)
          or class Container(containers.DeclarativeContainer):
              payment_gateway = providers.Factory(StripeGateway)
        Returns (provider_name, token_name)
        """
        text = code.decode("utf-8", errors="ignore")
        import re

        # DeclarativeContainer pattern
        decl = re.findall(r"(\w+)\s*=\s*providers\.\w+\(([\w\.]+)\)", text)
        for token, impl in decl:
            yield impl, token

    def _text(self, code: bytes, node) -> str:
        return code[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
