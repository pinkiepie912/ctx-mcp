"""
Enhanced PyGraphBuilder with storage, caching, and real-time updates.

This module extends the original PyGraphBuilder with Phase 2 functionality:
- Persistent storage with incremental updates
- Multi-level caching for performance
- Real-time file monitoring
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from tree_sitter import Node as TsNode

from core.parsers.edge import EdgeParser
from core.parsers.node import NodeParser

from .cache import CacheManager
from .models import DiGraph, Edge, Node
from .storage import GraphStorage
from .watcher import FileWatcher

logger = logging.getLogger(__name__)


class EnhancedPyGraphBuilder:
    """
    Enhanced PyGraphBuilder with storage, caching, and real-time monitoring.

    This builder extends the original functionality with:
    - Persistent graph storage using JSONL format
    - Multi-level caching for ASTs and analysis results
    - Real-time file monitoring for incremental updates
    - Performance optimization and memory management
    """

    def __init__(
        self, root: Path, enable_watching: bool = False, enable_caching: bool = True
    ):
        """
        Initialize the enhanced graph builder.

        Args:
            root: Root directory of the project
            enable_watching: Enable real-time file monitoring
            enable_caching: Enable multi-level caching
        """
        self.root = root
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)

        # Initialize storage and cache systems
        self.storage = GraphStorage(root)
        self.cache_manager = CacheManager(root) if enable_caching else None

        # Initialize file watcher if requested
        self.watcher: Optional[FileWatcher] = None
        if enable_watching:
            self.watcher = FileWatcher(
                project_root=root,
                storage=self.storage,
                file_processor=self._process_single_file,
                ignore_patterns={
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    ".mypy_cache",
                    "node_modules",
                    ".venv",
                    "venv",
                    ".env",
                    ".ctx_mcp",
                },
            )

        self._is_watching = False

    async def build(self, force_rebuild: bool = False) -> DiGraph:
        """
        Build a complete dependency graph from Python source files.

        Args:
            force_rebuild: If True, ignore cached data and rebuild from scratch

        Returns:
            DiGraph containing all nodes and edges found in the codebase
        """
        # Try to load existing graph if not forcing rebuild
        if not force_rebuild:
            existing_graph = await self.storage.load_graph()
            if existing_graph is not None:
                logger.info("Loaded existing graph from storage")
                return existing_graph

        logger.info("Building dependency graph from scratch")
        nodes: Dict[str, Node] = {}
        edges: List[Edge] = []

        # Get all Python files, respecting ignore patterns
        python_files = self._get_python_files()

        total_files = len(python_files)
        processed_files = 0

        for file in python_files:
            try:
                # Convert to relative path for storage
                relative_path = str(file.relative_to(self.root))

                # Check if file needs processing
                file_hash = self.storage.get_file_hash(file)
                if not force_rebuild and not self.storage.needs_update(
                    relative_path, file_hash
                ):
                    logger.debug(f"Skipping unchanged file: {relative_path}")
                    continue

                # Parse the file
                file_nodes, file_edges = await self._process_file_async(
                    file, relative_path, file_hash
                )

                # Add nodes to the collection
                for node in file_nodes:
                    nodes[node.id] = node

                # Add edges to the collection
                edges.extend(file_edges)

                processed_files += 1
                if processed_files % 10 == 0:
                    logger.info(f"Processed {processed_files}/{total_files} files")

            except FileNotFoundError:
                logger.warning(f"File not found during parsing: {file}")
                continue
            except PermissionError:
                logger.warning(f"Permission denied reading file: {file}")
                continue
            except UnicodeDecodeError:
                logger.warning(f"Unicode decode error in file: {file}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error parsing file {file}: {e}", exc_info=True)
                continue

        # Create complete dependency graph
        graph = DiGraph(
            nodes=list(nodes.values()),
            edges=edges,
            bindings={},  # Environment-specific bindings can be added later
        )

        # Save the graph
        await self.storage.save_graph(graph)
        logger.info(f"Built graph with {len(nodes)} nodes and {len(edges)} edges")

        return graph

    async def _process_file_async(
        self, file_path: Path, relative_path: str, file_hash: str
    ) -> Tuple[List[Node], List[Edge]]:
        """
        Process a single file asynchronously with caching support.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from project root
            file_hash: Content hash of the file

        Returns:
            Tuple of (nodes, edges) found in the file
        """
        # Check cache first
        if self.cache_manager:
            cached_result = self.cache_manager.get_ast_cache(relative_path, file_hash)
            if cached_result:
                logger.debug(f"Using cached parse result for {relative_path}")
                return cached_result

        # Read file content
        try:
            code = file_path.read_bytes()
            if not code.strip():
                return [], []

            # Cache file content
            if self.cache_manager:
                self.cache_manager.cache_file_content(relative_path, code)

            # Parse the file
            tree = self.parser.parse(code)
            nodes, edges = self._parse_file(code, tree.root_node, relative_path)

            # Cache the result
            if self.cache_manager:
                self.cache_manager.cache_ast(relative_path, file_hash, (nodes, edges))

            return nodes, edges

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return [], []

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
            except Exception as e:
                # Continue parsing even if edge parsing fails for this node
                logger.debug(f"Edge parsing failed for node in {filepath}: {e}")
                pass

            # Add children to stack in reverse order to maintain left-to-right processing
            for child in reversed(current_ts_node.children):
                traversal_stack.append((child, current_parent))

        return nodes, edges

    def _process_single_file(self, filepath: str) -> Tuple[List[Node], List[Edge]]:
        """
        Synchronous wrapper for file processing (used by FileWatcher).

        Args:
            filepath: Absolute path to the file

        Returns:
            Tuple of (nodes, edges) found in the file
        """
        try:
            file_path = Path(filepath)
            relative_path = str(file_path.relative_to(self.root))
            file_hash = self.storage.get_file_hash(file_path)

            # Try to use existing event loop first
            try:
                loop = asyncio.get_running_loop()
                # If we're in an event loop, use asyncio.create_task or run in thread
                # For now, let's use the synchronous version to avoid loop conflicts
                return self._process_file_sync(file_path, relative_path, file_hash)
            except RuntimeError:
                # No running event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._process_file_async(file_path, relative_path, file_hash)
                    )
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

        except Exception as e:
            logger.error(f"Error in synchronous file processing for {filepath}: {e}")
            return [], []

    def _process_file_sync(
        self, file_path: Path, relative_path: str, file_hash: str
    ) -> Tuple[List[Node], List[Edge]]:
        """
        Synchronous version of file processing without async/await.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from project root
            file_hash: Content hash of the file

        Returns:
            Tuple of (nodes, edges) found in the file
        """
        # Check cache first (synchronous version)
        if self.cache_manager:
            cached_result = self.cache_manager.get_ast_cache(relative_path, file_hash)
            if cached_result:
                logger.debug(f"Using cached parse result for {relative_path}")
                return cached_result

        # Read file content
        try:
            code = file_path.read_bytes()
            if not code.strip():
                return [], []

            # Cache file content
            if self.cache_manager:
                self.cache_manager.cache_file_content(relative_path, code)

            # Parse the file
            tree = self.parser.parse(code)
            nodes, edges = self._parse_file(code, tree.root_node, relative_path)

            # Cache the result
            if self.cache_manager:
                self.cache_manager.cache_ast(relative_path, file_hash, (nodes, edges))

            return nodes, edges

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return [], []

    def _get_python_files(self) -> List[Path]:
        """
        Get all Python files in the project, respecting ignore patterns.

        Returns:
            List of Python file paths
        """
        ignore_patterns = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            ".ctx_mcp",
        }

        python_files = []
        for file in self.root.rglob("*.py"):
            # Check if file should be ignored using path parts
            should_ignore = False
            file_parts = file.parts
            for pattern in ignore_patterns:
                if pattern in file_parts:
                    should_ignore = True
                    break

            if not should_ignore:
                python_files.append(file)

        return python_files

    async def start_watching(self) -> None:
        """
        Start real-time file monitoring.

        This will begin monitoring the project directory for changes
        and updating the dependency graph incrementally.
        """
        if self.watcher is None:
            raise RuntimeError(
                "FileWatcher not initialized. Set enable_watching=True in constructor."
            )

        if self._is_watching:
            logger.warning("File watcher is already running")
            return

        await self.watcher.start_watching()
        self._is_watching = True
        logger.info("Started file watching for real-time updates")

    async def stop_watching(self) -> None:
        """
        Stop real-time file monitoring.
        """
        if self.watcher is None or not self._is_watching:
            return

        await self.watcher.stop_watching()
        self._is_watching = False
        logger.info("Stopped file watching")

    async def get_file_dependencies(self, filepath: str) -> Dict[str, List[str]]:
        """
        Get dependencies for a specific file.

        Args:
            filepath: Path to the file (relative to project root)

        Returns:
            Dictionary with direct and transitive dependencies
        """
        return await self.storage.get_file_dependencies(filepath)

    async def invalidate_file_cache(self, filepath: str) -> None:
        """
        Invalidate cached data for a specific file.

        Args:
            filepath: Path to the file (relative to project root)
        """
        if self.cache_manager:
            self.cache_manager.invalidate_file(filepath)

    def get_cache_stats(self) -> Optional[Dict[str, any]]:
        """
        Get cache statistics.

        Returns:
            Cache statistics or None if caching is disabled
        """
        if self.cache_manager:
            return self.cache_manager.get_cache_stats()
        return None

    async def optimize_performance(self) -> None:
        """
        Perform performance optimization (cache cleanup, etc.).
        """
        if self.cache_manager:
            await self.cache_manager.optimize_caches()

    async def force_update_file(self, filepath: str) -> None:
        """
        Force an update for a specific file.

        Args:
            filepath: Path to the file (absolute or relative to project root)
        """
        if self.watcher:
            await self.watcher.force_update_file(filepath)
        else:
            # Manual update without watcher
            try:
                file_path = Path(filepath)
                if not file_path.is_absolute():
                    file_path = self.root / filepath

                if file_path.exists():
                    relative_path = str(file_path.relative_to(self.root))
                    file_hash = self.storage.get_file_hash(file_path)
                    nodes, edges = await self._process_file_async(
                        file_path, relative_path, file_hash
                    )
                    await self.storage.update_file_data(
                        relative_path, nodes, edges, file_hash
                    )
                    logger.info(f"Force updated file: {relative_path}")
                else:
                    relative_path = str(Path(filepath).relative_to(self.root))
                    await self.storage.remove_file_data(relative_path)
                    logger.info(f"Removed deleted file: {relative_path}")

            except Exception as e:
                logger.error(f"Error force updating file {filepath}: {e}")

    def get_monitoring_stats(self) -> Optional[Dict[str, any]]:
        """
        Get file monitoring statistics.

        Returns:
            Monitoring statistics or None if watching is disabled
        """
        if self.watcher:
            try:
                # Try to get running loop first
                loop = asyncio.get_running_loop()
                # If in a loop, return a task or use synchronous version
                # For now, let's use synchronous version to avoid conflicts
                return self.watcher.get_monitoring_stats_sync()
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                return asyncio.run(self.watcher.get_monitoring_stats())
        return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._is_watching:
            asyncio.run(self.stop_watching())


# Backward compatibility alias
PyGraphBuilder = EnhancedPyGraphBuilder

