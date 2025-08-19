"""
Graph storage system with incremental updates and persistence.

This module provides persistent storage for dependency graphs using JSONL format
for efficient incremental updates and recovery capabilities.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from .models import DiGraph, Edge, Node

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata for a single file in the project."""

    filepath: str
    last_modified: datetime
    content_hash: str
    node_count: int
    edge_count: int


@dataclass
class GraphMetadata:
    """Metadata for the entire graph."""

    version: str
    created_at: datetime
    last_updated: datetime
    total_files: int
    total_nodes: int
    total_edges: int
    file_metadata: Dict[str, FileMetadata]


class GraphStorage:
    """
    Persistent graph storage with incremental updates.

    Uses JSONL format for efficient partial updates and provides
    hash-based change detection to minimize unnecessary work.
    """

    def __init__(self, project_root: Path):
        """
        Initialize storage for a project.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.storage_dir = project_root / ".ctx_mcp"
        self.graph_file = self.storage_dir / "graph.jsonl"
        self.metadata_file = self.storage_dir / "metadata.json"
        self.temp_file = self.storage_dir / "graph.jsonl.tmp"

        # Ensure storage directory exists
        self.storage_dir.mkdir(exist_ok=True)

        # In-memory cache
        self._graph_cache: Optional[DiGraph] = None
        self._metadata_cache: Optional[GraphMetadata] = None
        self._file_hashes: Dict[str, str] = {}
        self._node_to_file_cache: Optional[Dict[str, str]] = None

    async def save_graph(self, graph: DiGraph) -> None:
        """
        Save complete graph with metadata.

        Args:
            graph: Complete dependency graph to save
        """
        try:
            # Write to temporary file first for atomic operation
            await self._write_graph_to_file(graph, self.temp_file)

            # Update metadata
            metadata = await self._create_metadata(graph)
            await self._save_metadata(metadata)

            # Atomic rename
            self.temp_file.rename(self.graph_file)

            # Update caches
            self._graph_cache = graph
            self._metadata_cache = metadata
            self._node_to_file_cache = None  # Invalidate node-to-file cache

            # Update file hashes from metadata
            if metadata and metadata.file_metadata:
                for filepath, file_meta in metadata.file_metadata.items():
                    if file_meta.content_hash:
                        self._file_hashes[filepath] = file_meta.content_hash

            logger.info(
                f"Saved graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
            )

        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            if self.temp_file.exists():
                self.temp_file.unlink()
            raise

    async def load_graph(self) -> Optional[DiGraph]:
        """
        Load complete graph from storage.

        Returns:
            Complete dependency graph or None if not found
        """
        if self._graph_cache is not None:
            return self._graph_cache

        if not self.graph_file.exists():
            logger.info("No existing graph found")
            return None

        try:
            graph = await self._load_graph_from_file(self.graph_file)
            metadata = await self._load_metadata()

            # Update caches
            self._graph_cache = graph
            self._metadata_cache = metadata
            self._node_to_file_cache = None  # Invalidate node-to-file cache

            if metadata:
                self._file_hashes = {
                    fm.filepath: fm.content_hash
                    for fm in metadata.file_metadata.values()
                }

            logger.info(
                f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
            )
            return graph

        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return None

    async def update_file_data(
        self, filepath: str, nodes: List[Node], edges: List[Edge], file_hash: str
    ) -> None:
        """
        Incremental update for single file changes.

        Args:
            filepath: Path to the changed file
            nodes: New nodes from the file
            edges: New edges from the file
            file_hash: Content hash of the file
        """
        # Load existing graph
        graph = await self.load_graph()
        if graph is None:
            # No existing graph, create new one
            graph = DiGraph(nodes=nodes, edges=edges)
            # Update file hash before saving new graph
            self._file_hashes[filepath] = file_hash
            await self.save_graph(graph)
            return

        # Remove old nodes and edges for this file
        graph.nodes = [n for n in graph.nodes if n.filepath != filepath]
        graph.edges = [
            e for e in graph.edges if not self._edge_belongs_to_file(e, filepath)
        ]

        # Add new nodes and edges
        graph.nodes.extend(nodes)
        graph.edges.extend(edges)

        # Update file hash before saving
        self._file_hashes[filepath] = file_hash

        # Save updated graph
        await self.save_graph(graph)

        logger.info(f"Updated file {filepath}: {len(nodes)} nodes, {len(edges)} edges")

    async def remove_file_data(self, filepath: str) -> None:
        """
        Remove all data for a deleted file.

        Args:
            filepath: Path to the deleted file
        """
        graph = await self.load_graph()
        if graph is None:
            return

        # Remove nodes and edges for this file
        old_node_count = len(graph.nodes)
        old_edge_count = len(graph.edges)

        graph.nodes = [n for n in graph.nodes if n.filepath != filepath]
        graph.edges = [
            e for e in graph.edges if not self._edge_belongs_to_file(e, filepath)
        ]

        # Remove from hash tracking
        self._file_hashes.pop(filepath, None)

        # Save updated graph
        await self.save_graph(graph)

        removed_nodes = old_node_count - len(graph.nodes)
        removed_edges = old_edge_count - len(graph.edges)
        logger.info(
            f"Removed file {filepath}: {removed_nodes} nodes, {removed_edges} edges"
        )

    async def get_file_dependencies(self, filepath: str) -> Dict[str, List[str]]:
        """
        Get direct and transitive dependencies for a file.

        Args:
            filepath: Path to the file

        Returns:
            Dictionary with 'direct' and 'transitive' dependency lists
        """
        graph = await self.load_graph()
        if graph is None:
            return {"direct": [], "transitive": []}

        # Build adjacency map for efficient traversal
        adjacency: Dict[str, Set[str]] = {}
        for edge in graph.edges:
            if edge.source not in adjacency:
                adjacency[edge.source] = set()
            adjacency[edge.source].add(edge.target)

        # Get nodes in the file
        file_nodes = [n.id for n in graph.nodes if n.filepath == filepath]

        # Find direct dependencies
        direct_deps = set()
        for node_id in file_nodes:
            if node_id in adjacency:
                direct_deps.update(adjacency[node_id])

        # Find transitive dependencies using BFS
        transitive_deps = set(direct_deps)
        queue = list(direct_deps)
        visited = set(direct_deps)

        while queue:
            current = queue.pop(0)
            if current in adjacency:
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        transitive_deps.add(neighbor)
                        queue.append(neighbor)

        # Convert to file paths using cached mapping
        node_to_file = self._get_node_to_file_mapping(graph)
        direct_files = list(
            {node_to_file.get(dep) for dep in direct_deps if node_to_file.get(dep)}
        )
        transitive_files = list(
            {node_to_file.get(dep) for dep in transitive_deps if node_to_file.get(dep)}
        )

        return {"direct": direct_files, "transitive": transitive_files}

    def _get_node_to_file_mapping(self, graph: DiGraph) -> Dict[str, str]:
        """
        Get cached node-to-file mapping.
        
        Args:
            graph: The graph to build mapping from
            
        Returns:
            Dictionary mapping node IDs to file paths
        """
        if self._node_to_file_cache is None:
            self._node_to_file_cache = {n.id: n.filepath for n in graph.nodes}
        return self._node_to_file_cache

    def needs_update(self, filepath: str, current_hash: str) -> bool:
        """
        Check if file needs reparsing based on hash.

        Args:
            filepath: Path to the file
            current_hash: Current content hash of the file

        Returns:
            True if file needs to be reparsed
        """
        stored_hash = self._file_hashes.get(filepath)
        return stored_hash != current_hash

    def get_file_hash(self, filepath: Path) -> str:
        """
        Calculate content hash for a file.

        Args:
            filepath: Path to the file

        Returns:
            SHA-256 hash of file content
        """
        try:
            content = filepath.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash file {filepath}: {e}")
            return ""

    async def _write_graph_to_file(self, graph: DiGraph, filepath: Path) -> None:
        """Write graph to JSONL file."""
        with open(filepath, "w", encoding="utf-8") as f:
            # Write metadata line
            f.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "version": graph.version,
                        "env": graph.env,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                + "\n"
            )

            # Write nodes
            for node in graph.nodes:
                f.write(
                    json.dumps({"type": "node", "data": node.model_dump(mode="json")})
                    + "\n"
                )

            # Write edges
            for edge in graph.edges:
                f.write(
                    json.dumps({"type": "edge", "data": edge.model_dump(mode="json")})
                    + "\n"
                )

            # Write bindings
            if graph.bindings:
                f.write(json.dumps({"type": "bindings", "data": graph.bindings}) + "\n")

    async def _load_graph_from_file(self, filepath: Path) -> DiGraph:
        """Load graph from JSONL file."""
        graph = DiGraph()

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    obj_type = obj.get("type")

                    if obj_type == "metadata":
                        graph.version = obj.get("version", "0.1.0")
                        graph.env = obj.get("env", "dev")
                    elif obj_type == "node":
                        node = Node(**obj["data"])
                        graph.nodes.append(node)
                    elif obj_type == "edge":
                        edge = Edge(**obj["data"])
                        graph.edges.append(edge)
                    elif obj_type == "bindings":
                        graph.bindings = obj["data"]

                except Exception as e:
                    logger.warning(f"Skipping malformed line in graph file: {e}")
                    continue

        return graph

    async def _create_metadata(self, graph: DiGraph) -> GraphMetadata:
        """Create metadata for the graph."""
        now = datetime.now(timezone.utc)

        # Start with tracked files from hashes
        files_data: Dict[str, FileMetadata] = {}
        for filepath, file_hash in self._file_hashes.items():
            files_data[filepath] = FileMetadata(
                filepath=filepath,
                last_modified=now,
                content_hash=file_hash,
                node_count=0,
                edge_count=0,
            )

        # Count nodes by file
        for node in graph.nodes:
            filepath = node.filepath
            if filepath not in files_data:
                file_hash = self._file_hashes.get(filepath, "")
                files_data[filepath] = FileMetadata(
                    filepath=filepath,
                    last_modified=now,
                    content_hash=file_hash,
                    node_count=0,
                    edge_count=0,
                )
            files_data[filepath].node_count += 1

        # Count edges by file using cached mapping
        node_to_file = self._get_node_to_file_mapping(graph)
        for edge in graph.edges:
            source_file = node_to_file.get(edge.source)
            if source_file and source_file in files_data:
                files_data[source_file].edge_count += 1

        return GraphMetadata(
            version=graph.version,
            created_at=now,
            last_updated=now,
            total_files=len(files_data),
            total_nodes=len(graph.nodes),
            total_edges=len(graph.edges),
            file_metadata=files_data,
        )

    async def _save_metadata(self, metadata: GraphMetadata) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            # Convert to dict for JSON serialization
            data = asdict(metadata)
            # Convert datetime objects to ISO strings
            data["created_at"] = metadata.created_at.isoformat()
            data["last_updated"] = metadata.last_updated.isoformat()

            for file_meta in data["file_metadata"].values():
                file_meta["last_modified"] = file_meta["last_modified"].isoformat()

            json.dump(data, f, indent=2)

    async def _load_metadata(self) -> Optional[GraphMetadata]:
        """Load metadata from file."""
        if not self.metadata_file.exists():
            return None

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert ISO strings back to datetime objects
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])

            file_metadata = {}
            for filepath, file_data in data["file_metadata"].items():
                file_data["last_modified"] = datetime.fromisoformat(
                    file_data["last_modified"]
                )
                file_metadata[filepath] = FileMetadata(**file_data)

            data["file_metadata"] = file_metadata
            return GraphMetadata(**data)

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None

    def _edge_belongs_to_file(self, edge: Edge, filepath: str) -> bool:
        """Check if an edge belongs to a specific file."""
        # Edge belongs to file if source node is from that file
        return edge.source.startswith(f"{filepath}::")

