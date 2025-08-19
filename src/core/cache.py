"""
Multi-level caching system for performance optimization.

This module provides efficient caching for parsed ASTs, dependency data,
and frequently accessed graph queries to minimize computational overhead.
"""

import hashlib
import logging
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        """Calculate size after initialization."""
        self.size_bytes = self._calculate_size()

    def _calculate_size(self) -> int:
        """Estimate memory size of the cached value."""
        try:
            if isinstance(self.value, (str, bytes)):
                return len(self.value)
            elif isinstance(self.value, (list, tuple)):
                return sum(len(str(item)) for item in self.value)
            elif isinstance(self.value, dict):
                return sum(len(str(k)) + len(str(v)) for k, v in self.value.items())
            else:
                # Rough estimate using pickle size
                return len(pickle.dumps(self.value))
        except Exception:
            # Fallback to a default estimate
            return 1024


class LRUCache:
    """
    Thread-safe LRU cache with size limits and TTL support.

    This cache automatically evicts least recently used items when
    size limits are exceeded and supports time-based expiration.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl_hours: int = 24,
    ):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl_hours: Default TTL for entries in hours
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = timedelta(hours=default_ttl_hours)

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._current_memory = 0

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check if expired
            if self._is_expired(entry):
                self._remove_entry(key)
                self._misses += 1
                return None

            # Update access info and move to end (most recent)
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._cache.move_to_end(key)

            self._hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """
        Put a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (uses default if None)
        """
        with self._lock:
            now = datetime.now()

            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Create new entry
            entry = CacheEntry(key=key, value=value, created_at=now, last_accessed=now)

            # Check if single entry is too large
            if entry.size_bytes > self.max_memory_bytes:
                logger.warning(
                    f"Cache entry {key} is too large ({entry.size_bytes} bytes), skipping"
                )
                return

            # Add to cache
            self._cache[key] = entry
            self._current_memory += entry.size_bytes

            # Evict if necessary
            self._evict_if_necessary()

    def remove(self, key: str) -> bool:
        """
        Remove a specific key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hit_rate": round(hit_rate, 2),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }

    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update memory tracking."""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_memory -= entry.size_bytes

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry has expired."""
        age = datetime.now() - entry.created_at
        return age > self.default_ttl

    def _evict_if_necessary(self) -> None:
        """Evict entries if size or memory limits are exceeded."""
        # Evict expired entries first
        expired_keys = [
            key for key, entry in self._cache.items() if self._is_expired(entry)
        ]
        for key in expired_keys:
            self._remove_entry(key)
            self._evictions += 1

        # Evict LRU entries if still over limits
        while (
            len(self._cache) > self.max_size
            or self._current_memory > self.max_memory_bytes
        ):
            if not self._cache:
                break

            # Remove least recently used (first item)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._evictions += 1


class CacheManager:
    """
    Multi-level cache manager for different types of data.

    Provides specialized caches for ASTs, file contents, dependency queries,
    and graph analysis results with appropriate eviction policies.
    """

    def __init__(self, project_root: Path):
        """
        Initialize the cache manager.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root

        # Specialized caches with different configurations
        self.ast_cache = LRUCache(
            max_size=500,
            max_memory_mb=50,
            default_ttl_hours=12,  # ASTs are expensive to parse
        )

        self.file_content_cache = LRUCache(
            max_size=1000,
            max_memory_mb=30,
            default_ttl_hours=6,  # File contents change more frequently
        )

        self.dependency_cache = LRUCache(
            max_size=2000,
            max_memory_mb=20,
            default_ttl_hours=24,  # Dependency queries are relatively stable
        )

        self.analysis_cache = LRUCache(
            max_size=100,
            max_memory_mb=40,
            default_ttl_hours=48,  # Analysis results are expensive to compute
        )

        # File hash tracking for invalidation
        self._file_hashes: Dict[str, str] = {}

    def get_file_content(self, filepath: str) -> Optional[bytes]:
        """
        Get cached file content.

        Args:
            filepath: Path to the file

        Returns:
            File content or None if not cached or invalid
        """
        # Check if file has changed
        current_hash = self._get_file_hash(filepath)
        cache_key = f"content:{filepath}"

        cached_entry = self.file_content_cache.get(cache_key)
        if cached_entry:
            cached_content, cached_hash = cached_entry
            if cached_hash == current_hash:
                return cached_content
            else:
                # File changed, remove from cache
                self.file_content_cache.remove(cache_key)

        return None

    def cache_file_content(self, filepath: str, content: bytes) -> None:
        """
        Cache file content with hash validation.

        Args:
            filepath: Path to the file
            content: File content to cache
        """
        file_hash = hashlib.sha256(content).hexdigest()
        cache_key = f"content:{filepath}"

        self.file_content_cache.put(cache_key, (content, file_hash))
        self._file_hashes[filepath] = file_hash

    def get_ast_cache(self, filepath: str, file_hash: str) -> Optional[Any]:
        """
        Get cached AST for a file.

        Args:
            filepath: Path to the file
            file_hash: Current hash of the file

        Returns:
            Cached AST or None if not cached or invalid
        """
        cache_key = f"ast:{filepath}:{file_hash}"
        return self.ast_cache.get(cache_key)

    def cache_ast(self, filepath: str, file_hash: str, ast: Any) -> None:
        """
        Cache AST for a file.

        Args:
            filepath: Path to the file
            file_hash: Hash of the file content
            ast: Parsed AST to cache
        """
        cache_key = f"ast:{filepath}:{file_hash}"
        self.ast_cache.put(cache_key, ast)

    def get_dependency_query(self, query_key: str) -> Optional[Any]:
        """
        Get cached dependency query result.

        Args:
            query_key: Unique key for the query

        Returns:
            Cached result or None if not cached
        """
        return self.dependency_cache.get(query_key)

    def cache_dependency_query(self, query_key: str, result: Any) -> None:
        """
        Cache dependency query result.

        Args:
            query_key: Unique key for the query
            result: Query result to cache
        """
        self.dependency_cache.put(query_key, result)

    def get_analysis_result(self, analysis_key: str) -> Optional[Any]:
        """
        Get cached analysis result.

        Args:
            analysis_key: Unique key for the analysis

        Returns:
            Cached result or None if not cached
        """
        return self.analysis_cache.get(analysis_key)

    def cache_analysis_result(self, analysis_key: str, result: Any) -> None:
        """
        Cache analysis result.

        Args:
            analysis_key: Unique key for the analysis
            result: Analysis result to cache
        """
        self.analysis_cache.put(analysis_key, result)

    def invalidate_file(self, filepath: str) -> None:
        """
        Invalidate all caches related to a specific file.

        Args:
            filepath: Path to the file that changed
        """
        # Remove file content cache
        content_key = f"content:{filepath}"
        self.file_content_cache.remove(content_key)

        # Remove AST caches (all hashes for this file)
        ast_keys_to_remove = [
            key
            for key in self.ast_cache._cache.keys()
            if key.startswith(f"ast:{filepath}:")
        ]
        for key in ast_keys_to_remove:
            self.ast_cache.remove(key)

        # Remove dependency queries that might be affected
        # This is a conservative approach - in practice, you might want
        # more sophisticated dependency tracking
        dep_keys_to_remove = [
            key for key in self.dependency_cache._cache.keys() if filepath in key
        ]
        for key in dep_keys_to_remove:
            self.dependency_cache.remove(key)

        # Remove related analysis results
        analysis_keys_to_remove = [
            key for key in self.analysis_cache._cache.keys() if filepath in key
        ]
        for key in analysis_keys_to_remove:
            self.analysis_cache.remove(key)

        # Update file hash tracking
        self._file_hashes.pop(filepath, None)

        logger.debug(f"Invalidated caches for {filepath}")

    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dictionary with stats for each cache type
        """
        return {
            "ast_cache": self.ast_cache.get_stats(),
            "file_content_cache": self.file_content_cache.get_stats(),
            "dependency_cache": self.dependency_cache.get_stats(),
            "analysis_cache": self.analysis_cache.get_stats(),
            "total_tracked_files": len(self._file_hashes),
        }

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.ast_cache.clear()
        self.file_content_cache.clear()
        self.dependency_cache.clear()
        self.analysis_cache.clear()
        self._file_hashes.clear()

        logger.info("Cleared all caches")

    def _get_file_hash(self, filepath: str) -> str:
        """
        Get current hash of a file.

        Args:
            filepath: Path to the file

        Returns:
            SHA-256 hash of file content
        """
        try:
            path = Path(filepath)
            if not path.exists():
                return ""

            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash file {filepath}: {e}")
            return ""

    async def optimize_caches(self) -> None:
        """
        Perform cache optimization (cleanup expired entries, etc.).

        This should be called periodically to maintain cache health.
        """
        logger.info("Starting cache optimization")

        # Force cleanup of expired entries
        for cache in [
            self.ast_cache,
            self.file_content_cache,
            self.dependency_cache,
            self.analysis_cache,
        ]:
            with cache._lock:
                cache._evict_if_necessary()

        # Log statistics after optimization
        stats = self.get_cache_stats()
        logger.info(f"Cache optimization complete: {stats}")

    def create_dependency_query_key(
        self, filepath: str, query_type: str, **params
    ) -> str:
        """
        Create a standardized key for dependency queries.

        Args:
            filepath: File being queried
            query_type: Type of query (e.g., 'direct_deps', 'transitive_deps')
            **params: Additional query parameters

        Returns:
            Standardized cache key
        """
        # Sort parameters for consistent key generation
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"dep:{query_type}:{filepath}:{param_str}"

    def create_analysis_key(self, analysis_type: str, scope: str, **params) -> str:
        """
        Create a standardized key for analysis results.

        Args:
            analysis_type: Type of analysis (e.g., 'complexity', 'security')
            scope: Scope of analysis (file path or 'project')
            **params: Additional analysis parameters

        Returns:
            Standardized cache key
        """
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"analysis:{analysis_type}:{scope}:{param_str}"

