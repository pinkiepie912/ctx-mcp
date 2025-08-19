from pathlib import Path
from typing import List, Optional

from tree_sitter import Node as TsNode

from core.models import AccessType, Edge, EdgeKind, InjectionType

from .text import parse_text

__all__ = ["EdgeParser"]


class EdgeParser:
    """
    Parser for extracting Edge relationships from Tree-sitter AST.

    This class identifies relationships between code elements including imports,
    function calls, inheritance, and dependency injection patterns.
    Uses cursor-based traversal consistent with NodeParser for performance.

    Attributes:
        code: Raw bytes of the source code
        filepath: Path to the source file
        module_path: Python module path derived from filepath

    Constants:
        ID_SEPARATOR: Separator used in node IDs (matches NodeParser)
        IMPORT_TYPES: Tree-sitter node types for import statements
        CALL_TYPES: Tree-sitter node types for function/method calls
        CLASS_DEF_TYPES: Tree-sitter node types for class definitions
    """

    # Class-level constants matching NodeParser pattern
    ID_SEPARATOR = "::"

    IMPORT_TYPES = frozenset({"import_statement", "import_from_statement"})

    CALL_TYPES = frozenset({"call", "attribute"})

    CLASS_DEF_TYPES = frozenset({"class_definition", "decorated_definition"})

    def __init__(self, code: bytes, filepath: str):
        """
        Initialize EdgeParser with source code and file path.

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

    def process_node(self, ts_node: TsNode, parent_id: Optional[str]) -> List[Edge]:
        """
        Process a Tree-sitter node and extract all relationship edges.

        This method analyzes a Tree-sitter node to identify various types of
        relationships including imports, calls, inheritance, and dependency injection.

        Args:
            ts_node: Tree-sitter node to process
            parent_id: ID of the parent node for context

        Returns:
            List of Edge objects representing relationships found in this node
        """
        edges: List[Edge] = []

        try:
            # Parse different types of relationships
            edges.extend(self._parse_imports(ts_node))
            edges.extend(self._parse_function_calls(ts_node, parent_id))
            edges.extend(self._parse_inheritance(ts_node, parent_id))
            edges.extend(self._parse_dependency_injection(ts_node, parent_id))
        except Exception:
            # Graceful degradation - continue parsing other relationships
            pass

        return edges

    def _get_module_path(self, filepath: str) -> str:
        """Convert file path to Python module path (matches NodeParser)"""
        path = Path(filepath)
        # Remove .py extension and convert path separators to dots
        module_parts = path.with_suffix("").parts

        # Remove common source directory prefixes dynamically
        common_prefixes = ["src", "lib", "source", "app", "code"]
        for prefix in common_prefixes:
            if prefix in module_parts:
                prefix_index = module_parts.index(prefix)
                module_parts = module_parts[prefix_index + 1 :]
                break

        return ".".join(module_parts) if module_parts else path.stem

    def _parse_imports(self, ts_node: TsNode) -> List[Edge]:
        """
        Parse import statements and create IMPORTS edges.

        Handles:
        - import module
        - from module import name
        - from .relative import name
        - import module as alias
        """
        edges: List[Edge] = []

        if ts_node.type not in self.IMPORT_TYPES:
            return edges

        try:
            if ts_node.type == "import_statement":
                edges.extend(self._parse_simple_import(ts_node))
            elif ts_node.type == "import_from_statement":
                edges.extend(self._parse_from_import(ts_node))
        except Exception:
            # Skip malformed import statements
            pass

        return edges

    def _parse_simple_import(self, ts_node: TsNode) -> List[Edge]:
        """Parse 'import module' statements"""
        edges: List[Edge] = []

        # Find dotted_name or aliased_import nodes
        for child in ts_node.children:
            if child.type in ("dotted_name", "aliased_import"):
                try:
                    module_name = self._extract_module_name(child)
                    if module_name:
                        source_id = f"{self.filepath}{self.ID_SEPARATOR}module{self.ID_SEPARATOR}{self.module_path}"
                        target_id = f"external{self.ID_SEPARATOR}module{self.ID_SEPARATOR}{module_name}"

                        edge = Edge(
                            kind=EdgeKind.IMPORTS,
                            source=source_id,
                            target=target_id,
                            source_line=(
                                ts_node.start_point[0] + 1,
                                ts_node.end_point[0] + 1,
                            ),
                            source_location=f"{self.filepath}:{ts_node.start_point[0] + 1}",
                            access_type=AccessType.IMPORT,
                            weight=1,
                        )
                        edges.append(edge)
                except Exception:
                    continue

        return edges

    def _parse_from_import(self, ts_node: TsNode) -> List[Edge]:
        """Parse 'from module import name' statements"""
        edges: List[Edge] = []

        try:
            module_name = None
            imported_names: List[str] = []

            # Track if we found the import keyword (to know we're past the module name)
            past_import_keyword = False

            # Extract module name and imported items
            for child in ts_node.children:
                if child.type == "import":
                    past_import_keyword = True
                elif (
                    child.type in ("dotted_name", "relative_module")
                    and not past_import_keyword
                ):
                    # This is the module being imported from
                    module_name = self._extract_module_name(child)
                elif child.type == "dotted_name" and past_import_keyword:
                    # These are the items being imported
                    item_name = self._extract_text_safe(child)
                    if item_name:
                        imported_names.append(item_name)
                elif child.type == "import_list":
                    imported_names = self._extract_import_list(child)

            if module_name:
                source_id = f"{self.filepath}{self.ID_SEPARATOR}module{self.ID_SEPARATOR}{self.module_path}"
                target_id = (
                    f"external{self.ID_SEPARATOR}module{self.ID_SEPARATOR}{module_name}"
                )

                edge = Edge(
                    kind=EdgeKind.IMPORTS,
                    source=source_id,
                    target=target_id,
                    source_line=(ts_node.start_point[0] + 1, ts_node.end_point[0] + 1),
                    source_location=f"{self.filepath}:{ts_node.start_point[0] + 1}",
                    access_type=AccessType.IMPORT,
                    weight=2,  # from imports typically have higher dependency weight
                )
                edges.append(edge)

        except Exception:
            pass

        return edges

    def _parse_function_calls(
        self, ts_node: TsNode, parent_id: Optional[str]
    ) -> List[Edge]:
        """
        Parse function and method calls to create CALLS edges.

        Handles:
        - function()
        - object.method()
        - Class()
        - await async_function()
        """
        edges: List[Edge] = []

        if ts_node.type != "call":
            return edges

        try:
            # Extract the function being called
            if ts_node.children:
                function_node = ts_node.children[
                    0
                ]  # First child is the function reference

                call_info = self._analyze_function_call(function_node)
                if call_info:
                    source_id = (
                        parent_id
                        or f"{self.filepath}{self.ID_SEPARATOR}module{self.ID_SEPARATOR}{self.module_path}"
                    )

                    edge = Edge(
                        kind=EdgeKind.CALLS,
                        source=source_id,
                        target=call_info["target"],
                        source_line=(
                            ts_node.start_point[0] + 1,
                            ts_node.end_point[0] + 1,
                        ),
                        source_location=f"{self.filepath}:{ts_node.start_point[0] + 1}",
                        access_type=call_info["access_type"],
                        weight=1,
                        async_context=self._is_in_async_context(ts_node),
                    )
                    edges.append(edge)

        except Exception:
            pass

        return edges

    def _parse_inheritance(
        self, ts_node: TsNode, parent_id: Optional[str]
    ) -> List[Edge]:
        """
        Parse class inheritance to create IMPLEMENTS edges.

        Handles:
        - class Child(Parent)
        - class Child(Parent1, Parent2)
        """
        edges: List[Edge] = []

        # Handle decorated class definitions
        actual_class_node = ts_node
        if ts_node.type == "decorated_definition":
            for child in ts_node.children:
                if child.type == "class_definition":
                    actual_class_node = child
                    break

        if actual_class_node.type != "class_definition":
            return edges

        try:
            class_name = None
            parent_classes: List[str] = []

            # Extract class name and parent classes
            for child in actual_class_node.children:
                if child.type == "identifier" and not class_name:
                    class_name = self._extract_text_safe(child)
                elif child.type == "argument_list":
                    parent_classes = self._extract_parent_classes(child)

            if class_name and parent_classes and parent_id:
                source_id = f"{self.filepath}{self.ID_SEPARATOR}class{self.ID_SEPARATOR}{class_name}"

                for parent_class in parent_classes:
                    if parent_class:
                        target_id = f"external{self.ID_SEPARATOR}class{self.ID_SEPARATOR}{parent_class}"

                        edge = Edge(
                            kind=EdgeKind.IMPLEMENTS,
                            source=source_id,
                            target=target_id,
                            source_line=(
                                actual_class_node.start_point[0] + 1,
                                actual_class_node.end_point[0] + 1,
                            ),
                            source_location=f"{self.filepath}:{actual_class_node.start_point[0] + 1}",
                            access_type=AccessType.INHERITANCE,
                            weight=3,  # Inheritance is a strong relationship
                        )
                        edges.append(edge)

        except Exception:
            pass

        return edges

    def _parse_dependency_injection(
        self, ts_node: TsNode, parent_id: Optional[str]
    ) -> List[Edge]:
        """
        Parse constructor dependency injection to create INJECTS edges.

        Handles:
        - def __init__(self, dependency: Type)
        - def __init__(self, dependency: Type = None)
        """
        edges: List[Edge] = []

        if ts_node.type not in ("function_definition", "async_function_definition"):
            return edges

        try:
            # Check if this is a constructor (__init__)
            function_name = None
            for child in ts_node.children:
                if child.type == "identifier":
                    function_name = self._extract_text_safe(child)
                    break

            if function_name != "__init__":
                return edges

            # Extract typed parameters (excluding self)
            parameters_node = None
            for child in ts_node.children:
                if child.type == "parameters":
                    parameters_node = child
                    break

            if not parameters_node or not parent_id:
                return edges

            dependencies = self._extract_dependencies(parameters_node)
            source_id = parent_id  # The class containing this constructor

            for dep_info in dependencies:
                if dep_info["type_hint"]:
                    target_id = f"external{self.ID_SEPARATOR}class{self.ID_SEPARATOR}{dep_info['type_hint']}"

                    edge = Edge(
                        kind=EdgeKind.INJECTS,
                        source=source_id,
                        target=target_id,
                        source_line=(
                            ts_node.start_point[0] + 1,
                            ts_node.end_point[0] + 1,
                        ),
                        source_location=f"{self.filepath}:{ts_node.start_point[0] + 1}",
                        access_type=AccessType.DIRECT,
                        injection_type=InjectionType.CONSTRUCTOR,
                        optional=dep_info["has_default"],
                        weight=2,
                    )
                    edges.append(edge)

        except Exception:
            pass

        return edges

    # Helper methods
    def _extract_module_name(self, node: TsNode) -> Optional[str]:
        """Extract module name from dotted_name or relative_module node"""
        try:
            if node.type == "relative_module":
                # Handle relative imports like .module or ..module
                return self._extract_text_safe(node)
            elif node.type in ("dotted_name", "identifier", "aliased_import"):
                return self._extract_text_safe(node).split(" as ")[0]  # Handle aliases
        except Exception:
            pass
        return None

    def _extract_import_list(self, node: TsNode) -> List[str]:
        """Extract list of imported names from import_list node"""
        names: List[str] = []
        try:
            for child in node.children:
                if child.type in ("identifier", "aliased_import", "dotted_name"):
                    name = self._extract_text_safe(child)
                    if name:
                        names.append(name.split(" as ")[0])  # Handle aliases
        except Exception:
            pass
        return names

    def _analyze_function_call(self, node: TsNode) -> Optional[dict]:
        """Analyze function call to determine target and access type"""
        try:
            if node.type == "identifier":
                # Simple function call: func()
                func_name = self._extract_text_safe(node)
                return {
                    "target": f"external{self.ID_SEPARATOR}function{self.ID_SEPARATOR}{func_name}",
                    "access_type": AccessType.DIRECT,
                }
            elif node.type == "attribute":
                # Method call: obj.method()
                attr_text = self._extract_text_safe(node)
                return {
                    "target": f"external{self.ID_SEPARATOR}method{self.ID_SEPARATOR}{attr_text}",
                    "access_type": AccessType.ATTRIBUTE,
                }
        except Exception:
            pass
        return None

    def _extract_parent_classes(self, node: TsNode) -> List[str]:
        """Extract parent class names from argument_list"""
        parents: List[str] = []
        try:
            for child in node.children:
                if child.type in ("identifier", "dotted_name"):
                    parent = self._extract_text_safe(child)
                    if parent:
                        parents.append(parent)
        except Exception:
            pass
        return parents

    def _extract_dependencies(self, parameters_node: TsNode) -> List[dict]:
        """Extract dependency injection information from constructor parameters"""
        dependencies: List[dict] = []
        try:
            for child in parameters_node.children:
                if child.type in ("typed_parameter", "typed_default_parameter"):
                    dep_info = self._parse_dependency_parameter(child)
                    if dep_info and dep_info["name"] != "self":  # Skip self parameter
                        dependencies.append(dep_info)
        except Exception:
            pass
        return dependencies

    def _parse_dependency_parameter(self, node: TsNode) -> Optional[dict]:
        """Parse a single dependency parameter"""
        try:
            param_name = None
            type_hint = None
            has_default = node.type == "typed_default_parameter"

            for child in node.children:
                if child.type == "identifier":
                    param_name = self._extract_text_safe(child)
                elif child.type == "type":
                    type_hint = self._extract_text_safe(child)

            if param_name:
                return {
                    "name": param_name,
                    "type_hint": type_hint,
                    "has_default": has_default,
                }
        except Exception:
            pass
        return None

    def _is_in_async_context(self, node: TsNode) -> bool:
        """Check if a node is within an async function or has await keyword"""
        # Simple heuristic - check if we're in async function or have await
        current = node
        while current and current.parent:
            if current.type == "async_function_definition":
                return True
            if current.type == "await" or (
                current.type == "identifier"
                and self._extract_text_safe(current) == "await"
            ):
                return True
            current = current.parent
        return False

    def _extract_text_safe(self, node: TsNode) -> str:
        """Safely extract text from Tree-sitter node"""
        try:
            return parse_text(self.code, node) or ""
        except Exception:
            return ""
