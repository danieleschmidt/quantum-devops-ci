"""
Intelligent caching system for quantum DevOps CI/CD.

This module provides multi-level caching for quantum circuit results,
backend configurations, and expensive computations to improve performance.
"""

import hashlib
import json
import pickle
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import weakref

from .exceptions import ConfigurationError


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).seconds > self.ttl_seconds
    
    def access(self):
        """Record access to this entry."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class MemoryCache:
    """In-memory cache with LRU eviction."""
    
    def __init__(self, 
                 max_size_mb: int = 100, 
                 max_entries: int = 1000,
                 default_ttl_seconds: Optional[int] = None):
        """
        Initialize memory cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of entries
            default_ttl_seconds: Default TTL for entries
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._current_size_bytes = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None
            
            entry.access()
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                # Can't serialize, don't cache
                return False
            
            # Check if it would exceed size limits
            if size_bytes > self.max_size_bytes:
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict entries if necessary
            self._evict_if_necessary(size_bytes)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.default_ttl_seconds
            )
            
            self._cache[key] = entry
            self._current_size_bytes += size_bytes
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'entries': len(self._cache),
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'evictions': self.evictions,
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'max_entries': self.max_entries
            }
    
    def _remove_entry(self, key: str):
        """Remove entry and update size."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_size_bytes -= entry.size_bytes
            del self._cache[key]
    
    def _evict_if_necessary(self, new_entry_size: int):
        """Evict entries using LRU policy if necessary."""
        # Check entry count limit
        while len(self._cache) >= self.max_entries:
            self._evict_lru_entry()
        
        # Check size limit
        while (self._current_size_bytes + new_entry_size) > self.max_size_bytes:
            if not self._evict_lru_entry():
                break  # No more entries to evict
    
    def _evict_lru_entry(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False
        
        # Find LRU entry
        lru_key = min(self._cache.keys(), 
                      key=lambda k: self._cache[k].last_accessed)
        
        self._remove_entry(lru_key)
        self.evictions += 1
        return True


class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 500):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        self._metadata_file = self.cache_dir / '_metadata.json'
        self._lock = threading.RLock()
        
        # Load existing metadata
        self._metadata = self._load_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            key_hash = self._hash_key(key)
            
            if key_hash not in self._metadata:
                return None
            
            entry_info = self._metadata[key_hash]
            
            # Check expiration
            if self._is_expired(entry_info):
                self.delete(key)
                return None
            
            # Load from disk
            file_path = self.cache_dir / f"{key_hash}.cache"
            if not file_path.exists():
                # Metadata inconsistent, remove entry
                del self._metadata[key_hash]
                self._save_metadata()
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access time
                entry_info['last_accessed'] = datetime.now().isoformat()
                entry_info['access_count'] += 1
                self._save_metadata()
                
                return value
            except Exception:
                # Corrupted file, remove
                self.delete(key)
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put value in disk cache."""
        with self._lock:
            key_hash = self._hash_key(key)
            file_path = self.cache_dir / f"{key_hash}.cache"
            
            # Remove existing entry if present
            if key_hash in self._metadata:
                self.delete(key)
            
            try:
                # Serialize and save
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                file_size = file_path.stat().st_size
                
                # Check if it would exceed size limit
                current_size = sum(info['size_bytes'] for info in self._metadata.values())
                if current_size + file_size > self.max_size_bytes:
                    # Try to evict old entries
                    self._evict_old_entries()
                    current_size = sum(info['size_bytes'] for info in self._metadata.values())
                    
                    if current_size + file_size > self.max_size_bytes:
                        # Still too big, don't cache
                        file_path.unlink()
                        return False
                
                # Add metadata
                now = datetime.now()
                self._metadata[key_hash] = {
                    'key': key,
                    'created_at': now.isoformat(),
                    'last_accessed': now.isoformat(),
                    'access_count': 0,
                    'size_bytes': file_size,
                    'expires_at': (now + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds else None
                }
                
                self._save_metadata()
                return True
                
            except Exception:
                # Clean up on failure
                if file_path.exists():
                    file_path.unlink()
                return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        with self._lock:
            key_hash = self._hash_key(key)
            
            if key_hash not in self._metadata:
                return False
            
            # Remove file
            file_path = self.cache_dir / f"{key_hash}.cache"
            if file_path.exists():
                file_path.unlink()
            
            # Remove metadata
            del self._metadata[key_hash]
            self._save_metadata()
            
            return True
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Remove all cache files
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
            
            # Clear metadata
            self._metadata.clear()
            self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(info['size_bytes'] for info in self._metadata.values())
            total_accesses = sum(info['access_count'] for info in self._metadata.values())
            
            return {
                'entries': len(self._metadata),
                'size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'total_accesses': total_accesses,
                'cache_dir': str(self.cache_dir)
            }
    
    def _hash_key(self, key: str) -> str:
        """Create hash for key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _is_expired(self, entry_info: Dict[str, Any]) -> bool:
        """Check if entry is expired."""
        expires_at = entry_info.get('expires_at')
        if not expires_at:
            return False
        
        return datetime.now() > datetime.fromisoformat(expires_at)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk."""
        if not self._metadata_file.exists():
            return {}
        
        try:
            with open(self._metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache metadata: {e}")
    
    def _evict_old_entries(self):
        """Evict old entries to make space."""
        if not self._metadata:
            return
        
        # Sort by last access time
        entries_by_access = sorted(
            self._metadata.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Evict oldest 25%
        num_to_evict = max(1, len(entries_by_access) // 4)
        
        for key_hash, _ in entries_by_access[:num_to_evict]:
            file_path = self.cache_dir / f"{key_hash}.cache"
            if file_path.exists():
                file_path.unlink()
            del self._metadata[key_hash]
        
        self._save_metadata()


class MultiLevelCache:
    """Multi-level cache with memory and disk tiers."""
    
    def __init__(self,
                 memory_cache_mb: int = 50,
                 disk_cache_mb: int = 200,
                 cache_dir: str = None,
                 default_ttl_seconds: Optional[int] = 3600):
        """
        Initialize multi-level cache.
        
        Args:
            memory_cache_mb: Memory cache size in MB
            disk_cache_mb: Disk cache size in MB
            cache_dir: Directory for disk cache
            default_ttl_seconds: Default TTL in seconds
        """
        self.memory_cache = MemoryCache(memory_cache_mb, default_ttl_seconds=default_ttl_seconds)
        
        if cache_dir is None:
            cache_dir = Path.home() / '.quantum_devops_ci' / 'cache'
        
        self.disk_cache = DiskCache(str(cache_dir), disk_cache_mb)
        self.default_ttl_seconds = default_ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (check memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.put(key, value, self.default_ttl_seconds)
            return value
        
        return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put value in cache (both memory and disk)."""
        ttl = ttl_seconds or self.default_ttl_seconds
        
        # Try to put in both caches
        memory_success = self.memory_cache.put(key, value, ttl)
        disk_success = self.disk_cache.put(key, value, ttl)
        
        return memory_success or disk_success
    
    def delete(self, key: str) -> bool:
        """Delete from both cache levels."""
        memory_deleted = self.memory_cache.delete(key)
        disk_deleted = self.disk_cache.delete(key)
        
        return memory_deleted or disk_deleted
    
    def clear(self):
        """Clear both cache levels."""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        
        return {
            'memory': memory_stats,
            'disk': disk_stats,
            'total_entries': memory_stats['entries'] + disk_stats['entries'],
            'total_size_mb': memory_stats['size_mb'] + disk_stats['size_mb']
        }


class CacheManager:
    """Central cache manager for quantum DevOps system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cache manager with configuration."""
        config = config or {}
        
        # Circuit result cache (longer TTL, larger size)
        self.circuit_cache = MultiLevelCache(
            memory_cache_mb=config.get('circuit_memory_mb', 100),
            disk_cache_mb=config.get('circuit_disk_mb', 500),
            cache_dir=config.get('circuit_cache_dir'),
            default_ttl_seconds=config.get('circuit_ttl_seconds', 7200)  # 2 hours
        )
        
        # Configuration cache (shorter TTL, smaller size)
        self.config_cache = MemoryCache(
            max_size_mb=config.get('config_memory_mb', 10),
            default_ttl_seconds=config.get('config_ttl_seconds', 300)  # 5 minutes
        )
        
        # Backend info cache (medium TTL)
        self.backend_cache = MultiLevelCache(
            memory_cache_mb=config.get('backend_memory_mb', 20),
            disk_cache_mb=config.get('backend_disk_mb', 50),
            default_ttl_seconds=config.get('backend_ttl_seconds', 1800)  # 30 minutes
        )
        
        # Cost calculation cache
        self.cost_cache = MemoryCache(
            max_size_mb=config.get('cost_memory_mb', 5),
            default_ttl_seconds=config.get('cost_ttl_seconds', 600)  # 10 minutes
        )
    
    def cache_circuit_result(self, circuit_hash: str, backend: str, 
                           shots: int, result: Any) -> bool:
        """Cache quantum circuit execution result."""
        key = f"circuit:{circuit_hash}:{backend}:{shots}"
        return self.circuit_cache.put(key, result)
    
    def get_circuit_result(self, circuit_hash: str, backend: str, shots: int) -> Optional[Any]:
        """Get cached circuit result."""
        key = f"circuit:{circuit_hash}:{backend}:{shots}"
        return self.circuit_cache.get(key)
    
    def cache_backend_info(self, backend_name: str, info: Any) -> bool:
        """Cache backend information."""
        key = f"backend_info:{backend_name}"
        return self.backend_cache.put(key, info)
    
    def get_backend_info(self, backend_name: str) -> Optional[Any]:
        """Get cached backend information."""
        key = f"backend_info:{backend_name}"
        return self.backend_cache.get(key)
    
    def cache_cost_estimate(self, job_spec_hash: str, estimate: Any) -> bool:
        """Cache cost estimate."""
        key = f"cost:{job_spec_hash}"
        return self.cost_cache.put(key, estimate)
    
    def get_cost_estimate(self, job_spec_hash: str) -> Optional[Any]:
        """Get cached cost estimate."""
        key = f"cost:{job_spec_hash}"
        return self.cost_cache.get(key)
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.circuit_cache.clear()
        self.config_cache.clear()
        self.backend_cache.clear()
        self.cost_cache.clear()
    
    async def optimize(self):
        """Optimize all caches by cleaning expired entries."""
        # This is a simple optimization - in practice could be more sophisticated
        pass
        
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            'circuit': self.circuit_cache.get_stats(),
            'config': self.config_cache.get_stats(),
            'backend': self.backend_cache.get_stats(),
            'cost': self.cost_cache.get_stats()
        }


def cached(cache_manager: CacheManager, cache_type: str, 
           key_func: Optional[Callable] = None, ttl_seconds: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache_manager: Cache manager instance
        cache_type: Type of cache to use ('circuit', 'config', 'backend', 'cost')
        key_func: Function to generate cache key from arguments
        ttl_seconds: TTL override for this cache
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Get appropriate cache
            if cache_type == 'circuit':
                cache = cache_manager.circuit_cache
            elif cache_type == 'config':
                cache = cache_manager.config_cache
            elif cache_type == 'backend':
                cache = cache_manager.backend_cache
            elif cache_type == 'cost':
                cache = cache_manager.cost_cache
            else:
                raise ConfigurationError(f"Unknown cache type: {cache_type}")
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            if ttl_seconds:
                if hasattr(cache, 'put'):
                    cache.put(cache_key, result, ttl_seconds)
                else:
                    # Memory cache doesn't support per-entry TTL
                    cache.put(cache_key, result)
            else:
                cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def create_circuit_hash(circuit) -> str:
    """Create hash for quantum circuit for caching."""
    try:
        # Try to get a string representation
        circuit_str = str(circuit)
        
        # Add more specific attributes if available
        if hasattr(circuit, 'num_qubits'):
            circuit_str += f"_qubits:{circuit.num_qubits}"
        
        if hasattr(circuit, 'depth'):
            circuit_str += f"_depth:{circuit.depth()}"
        
        return hashlib.md5(circuit_str.encode()).hexdigest()
    except Exception:
        # Fallback to object id
        return str(id(circuit))