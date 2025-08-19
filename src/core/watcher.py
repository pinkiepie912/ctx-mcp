"""
File system watcher for real-time dependency graph updates.

This module provides real-time monitoring of Python files and triggers
incremental graph updates when changes are detected.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .models import Edge, Node
from .storage import GraphStorage

logger = logging.getLogger(__name__)


@dataclass
class FileChangeEvent:
    """Represents a file change event with metadata."""

    filepath: str
    event_type: str  # 'modified', 'created', 'deleted', 'moved'
    timestamp: datetime
    old_filepath: Optional[str] = None  # For move events


class DebouncedEventProcessor:
    """
    Processes file events with debouncing to handle rapid successive changes.

    This prevents processing a file multiple times when editors make
    rapid changes (auto-save, etc.).
    """

    def __init__(self, debounce_delay: float = 0.1):
        """
        Initialize the debounced processor.

        Args:
            debounce_delay: Delay in seconds to wait for file stabilization
        """
        self.debounce_delay = debounce_delay
        self.pending_events: Dict[str, FileChangeEvent] = {}
        self.event_tasks: Dict[str, asyncio.Task] = {}
        self.processors: List[Callable[[FileChangeEvent], None]] = []

    def add_processor(self, processor: Callable[[FileChangeEvent], None]) -> None:
        """Add an event processor callback."""
        self.processors.append(processor)

    async def schedule_event(self, event: FileChangeEvent) -> None:
        """
        Schedule an event for processing with debouncing.

        Args:
            event: File change event to process
        """
        filepath = event.filepath

        # Cancel existing task for this file
        if filepath in self.event_tasks:
            self.event_tasks[filepath].cancel()

        # Store the latest event
        self.pending_events[filepath] = event

        # Schedule processing after debounce delay
        self.event_tasks[filepath] = asyncio.create_task(
            self._process_after_delay(filepath)
        )

    async def _process_after_delay(self, filepath: str) -> None:
        """Process an event after the debounce delay."""
        try:
            await asyncio.sleep(self.debounce_delay)

            # Get the event to process
            if filepath in self.pending_events:
                event = self.pending_events.pop(filepath)

                # Process with all registered processors
                for processor in self.processors:
                    try:
                        await processor(event)
                    except Exception as e:
                        logger.error(f"Error in event processor: {e}")

        except asyncio.CancelledError:
            # Task was cancelled, which is normal for debouncing
            pass
        except Exception as e:
            logger.error(f"Error processing debounced event for {filepath}: {e}")
        finally:
            # Clean up task reference
            self.event_tasks.pop(filepath, None)


class GraphUpdateHandler(FileSystemEventHandler):
    """
    Handles file system events and triggers graph updates.

    This handler filters for Python files and coordinates with the
    storage system to maintain an up-to-date dependency graph.
    """

    def __init__(
        self,
        storage: GraphStorage,
        file_processor: Callable[[str], tuple[List[Node], List[Edge]]],
        ignore_patterns: Set[str],
    ):
        """
        Initialize the graph update handler.

        Args:
            storage: Graph storage system
            file_processor: Function to parse a file and return nodes/edges
            ignore_patterns: Set of patterns to ignore (e.g., '.git', '__pycache__')
        """
        super().__init__()
        self.storage = storage
        self.file_processor = file_processor
        self.ignore_patterns = ignore_patterns
        self.event_processor = DebouncedEventProcessor()

        # Register ourselves as an event processor
        self.event_processor.add_processor(self._process_file_change)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory and self._should_process_file(event.src_path):
            asyncio.create_task(
                self.event_processor.schedule_event(
                    FileChangeEvent(
                        filepath=event.src_path,
                        event_type="modified",
                        timestamp=datetime.now(),
                    )
                )
            )

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory and self._should_process_file(event.src_path):
            asyncio.create_task(
                self.event_processor.schedule_event(
                    FileChangeEvent(
                        filepath=event.src_path,
                        event_type="created",
                        timestamp=datetime.now(),
                    )
                )
            )

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory and self._should_process_file(event.src_path):
            asyncio.create_task(
                self.event_processor.schedule_event(
                    FileChangeEvent(
                        filepath=event.src_path,
                        event_type="deleted",
                        timestamp=datetime.now(),
                    )
                )
            )

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events."""
        if not event.is_directory:
            old_path = event.src_path
            new_path = event.dest_path

            if self._should_process_file(old_path) or self._should_process_file(
                new_path
            ):
                asyncio.create_task(
                    self.event_processor.schedule_event(
                        FileChangeEvent(
                            filepath=new_path,
                            event_type="moved",
                            timestamp=datetime.now(),
                            old_filepath=old_path,
                        )
                    )
                )

    def _should_process_file(self, filepath: str) -> bool:
        """
        Check if a file should be processed.

        Args:
            filepath: Path to the file

        Returns:
            True if file should be processed
        """
        path = Path(filepath)

        # Only process Python files
        if path.suffix != ".py":
            return False

        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in str(path):
                return False

        return True

    async def _process_file_change(self, event: FileChangeEvent) -> None:
        """
        Process a file change event.

        Args:
            event: File change event to process
        """
        try:
            if event.event_type == "deleted":
                # Remove file from graph
                relative_path = self._get_relative_path(event.filepath)
                await self.storage.remove_file_data(relative_path)
                logger.info(f"Removed deleted file from graph: {relative_path}")

            elif event.event_type == "moved":
                # Handle file move
                if event.old_filepath:
                    old_relative = self._get_relative_path(event.old_filepath)
                    await self.storage.remove_file_data(old_relative)

                # Process new location if it's a Python file
                if self._should_process_file(event.filepath):
                    await self._process_file_update(event.filepath)

            else:  # created or modified
                await self._process_file_update(event.filepath)

        except Exception as e:
            logger.error(f"Error processing file change for {event.filepath}: {e}")

    async def _process_file_update(self, filepath: str) -> None:
        """
        Process a file update (creation or modification).

        Args:
            filepath: Path to the file that was updated
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"File no longer exists: {filepath}")
            return

        # Calculate file hash
        file_hash = self.storage.get_file_hash(path)
        relative_path = self._get_relative_path(filepath)

        # Check if file actually changed
        if not self.storage.needs_update(relative_path, file_hash):
            logger.debug(f"File {relative_path} has not changed (same hash)")
            return

        try:
            # Parse the file
            nodes, edges = self.file_processor(filepath)

            # Update storage
            await self.storage.update_file_data(relative_path, nodes, edges, file_hash)

            logger.info(
                f"Updated graph for {relative_path}: {len(nodes)} nodes, {len(edges)} edges"
            )

        except Exception as e:
            logger.error(f"Error parsing file {filepath}: {e}")

    def _get_relative_path(self, filepath: str) -> str:
        """Convert absolute path to relative path from project root."""
        try:
            return str(Path(filepath).relative_to(self.storage.project_root))
        except ValueError:
            # File is outside project root
            return filepath


class FileWatcher:
    """
    Real-time file system monitoring with batched updates.

    Monitors a project directory for changes to Python files and
    maintains an up-to-date dependency graph through incremental updates.
    """

    def __init__(
        self,
        project_root: Path,
        storage: GraphStorage,
        file_processor: Callable[[str], tuple[List[Node], List[Edge]]],
        ignore_patterns: Optional[Set[str]] = None,
    ):
        """
        Initialize the file watcher.

        Args:
            project_root: Root directory to monitor
            storage: Graph storage system
            file_processor: Function to parse files and extract nodes/edges
            ignore_patterns: Patterns to ignore during monitoring
        """
        self.project_root = project_root
        self.storage = storage
        self.file_processor = file_processor
        self.ignore_patterns = ignore_patterns or {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "node_modules",
            ".venv",
            "venv",
            ".env",
        }

        self.observer = Observer()
        self.handler = GraphUpdateHandler(
            storage=storage,
            file_processor=file_processor,
            ignore_patterns=self.ignore_patterns,
        )
        self.is_watching = False

    async def start_watching(self) -> None:
        """
        Start file system monitoring.

        This will begin monitoring the project directory for changes
        and updating the dependency graph in real-time.
        """
        if self.is_watching:
            logger.warning("File watcher is already running")
            return

        try:
            # Set up watchdog observer
            self.observer.schedule(self.handler, str(self.project_root), recursive=True)

            # Start monitoring
            self.observer.start()
            self.is_watching = True

            logger.info(f"Started watching {self.project_root} for changes")

        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            raise

    async def stop_watching(self) -> None:
        """
        Stop file system monitoring.

        This will stop monitoring for file changes and clean up resources.
        """
        if not self.is_watching:
            return

        try:
            self.observer.stop()
            self.observer.join(timeout=5.0)  # Wait up to 5 seconds
            self.is_watching = False

            logger.info("Stopped file watching")

        except Exception as e:
            logger.error(f"Error stopping file watcher: {e}")

    async def force_update_file(self, filepath: str) -> None:
        """
        Force an update for a specific file.

        Args:
            filepath: Path to the file to update
        """
        if Path(filepath).exists():
            await self.handler._process_file_update(filepath)
        else:
            relative_path = self.handler._get_relative_path(filepath)
            await self.storage.remove_file_data(relative_path)

    async def get_monitoring_stats(self) -> Dict[str, any]:
        """
        Get statistics about the monitoring system.

        Returns:
            Dictionary with monitoring statistics
        """
        return {
            "is_watching": self.is_watching,
            "project_root": str(self.project_root),
            "ignore_patterns": list(self.ignore_patterns),
            "pending_events": len(self.handler.event_processor.pending_events),
            "active_tasks": len(self.handler.event_processor.event_tasks),
        }

    def get_monitoring_stats_sync(self) -> Dict[str, any]:
        """
        Get statistics about the monitoring system (synchronous version).

        Returns:
            Dictionary with monitoring statistics
        """
        return {
            "is_watching": self.is_watching,
            "project_root": str(self.project_root),
            "ignore_patterns": list(self.ignore_patterns),
            "pending_events": len(self.handler.event_processor.pending_events),
            "active_tasks": len(self.handler.event_processor.event_tasks),
        }

    def __enter__(self):
        """Context manager entry."""
        asyncio.create_task(self.start_watching())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.create_task(self.stop_watching())

