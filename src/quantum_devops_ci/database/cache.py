"""
Caching layer for quantum DevOps CI/CD.

This module provides caching capabilities for expensive operations
like circuit compilation, backend status, and cost calculations.
"""

import json
import hashlib
import time
import warnings
from typing import Any, Optional, Dict, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import threading

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

T = TypeVar('T')


class CacheBackend(ABC):
    """Abstract cache backend."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL in seconds."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def keys(self, pattern: str = "*") -> list:
        """Get keys matching pattern."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize memory cache."""
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry['expires_at'] and datetime.now() > entry['expires_at']:
                del self._cache[key]
                return None
            
            # Update access time for LRU
            entry['accessed_at'] = datetime.now()
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL."""
        with self._lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()
            
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            self._cache[key] = {
                'value': value,
                'created_at': datetime.now(),
                'accessed_at': datetime.now(),
                'expires_at': expires_at
            }
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
    
    def clear(self) -> bool:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            return True
    
    def keys(self, pattern: str = "*") -> list:
        """Get keys matching pattern (simplified pattern matching)."""
        with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            
            # Simple prefix matching
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                return [key for key in self._cache.keys() if key.startswith(prefix)]
            
            return [key for key in self._cache.keys() if key == pattern]
    
    def _evict_oldest(self):
        """Evict oldest accessed entry."""
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]['accessed_at']
        )
        del self._cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_entries = len(self._cache)
            expired_entries = 0
            
            now = datetime.now()
            for entry in self._cache.values():
                if entry['expires_at'] and now > entry['expires_at']:
                    expired_entries += 1
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'max_size': self.max_size,
                'utilization': total_entries / self.max_size
            }


class FileCache(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize file cache."""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.quantum_devops_ci' / 'cache'
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash key to create safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check expiration
            if data['expires_at'] and datetime.now() > datetime.fromisoformat(data['expires_at']):
                file_path.unlink()
                return None
            
            return data['value']
            
        except (json.JSONDecodeError, KeyError, OSError):
            # Remove corrupted cache file
            try:
                file_path.unlink()
            except OSError:
                pass
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL."""
        file_path = self._get_file_path(key)
        
        expires_at = None
        if ttl:
            expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        
        data = {
            'key': key,
            'value': value,
            'created_at': datetime.now().isoformat(),
            'expires_at': expires_at
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, default=str)
            return True
        except (OSError, TypeError):
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        file_path = self._get_file_path(key)
        
        try:
            file_path.unlink()
            return True
        except OSError:
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
    
    def clear(self) -> bool:
        """Clear all cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            return True
        except OSError:
            return False
    
    def keys(self, pattern: str = "*") -> list:
        """Get keys matching pattern."""
        keys = []
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    key = data.get('key')
                    if key:
                        if pattern == "*" or key.startswith(pattern.rstrip("*")):
                            keys.append(key)
            except (json.JSONDecodeError, OSError):
                continue
        
        return keys


class RedisCache(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, **kwargs):
        """Initialize Redis cache."""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install redis-py: pip install redis")
        
        self.redis = redis.Redis(host=host, port=port, db=db, **kwargs)
        
        # Test connection
        try:
            self.redis.ping()
        except redis.ConnectionError:
            warnings.warn("Cannot connect to Redis server")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except (redis.RedisError, json.JSONDecodeError):
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL."""
        try:
            data = json.dumps(value, default=str)
            return self.redis.set(key, data, ex=ttl)
        except (redis.RedisError, TypeError):
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            return bool(self.redis.delete(key))
        except redis.RedisError:
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(self.redis.exists(key))
        except redis.RedisError:
            return False
    
    def clear(self) -> bool:
        """Clear all keys."""
        try:
            return bool(self.redis.flushdb())
        except redis.RedisError:
            return False
    
    def keys(self, pattern: str = "*") -> list:
        """Get keys matching pattern."""
        try:
            return [key.decode() for key in self.redis.keys(pattern)]
        except redis.RedisError:
            return []


class CacheManager:
    """Main cache manager with decorators and utilities."""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        """Initialize cache manager."""
        if backend is None:
            # Default to memory cache
            backend = MemoryCache()
        
        self.backend = backend
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.backend.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        return self.backend.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return self.backend.delete(key)
    
    def clear(self) -> bool:
        """Clear all cache."""
        return self.backend.clear()
    
    def cached(self, key: Optional[str] = None, ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key is None:
                    cache_key = self._generate_key(func.__name__, args, kwargs)
                else:
                    cache_key = key
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create deterministic key from function signature
        key_data = {
            'function': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def memoize(self, ttl: Optional[int] = None):
        """Memoization decorator (same as cached but clearer name)."""
        return self.cached(ttl=ttl)
    
    def cache_circuit_compilation(self, circuit_hash: str, backend: str, result: Any, ttl: int = 3600):
        """Cache compiled circuit results."""
        key = f"circuit_compilation:{circuit_hash}:{backend}"
        return self.set(key, result, ttl)
    
    def get_cached_compilation(self, circuit_hash: str, backend: str) -> Optional[Any]:
        """Get cached compilation result."""
        key = f"circuit_compilation:{circuit_hash}:{backend}"
        return self.get(key)
    
    def cache_backend_status(self, provider: str, backend: str, status: Dict[str, Any], ttl: int = 300):
        """Cache backend status information."""
        key = f"backend_status:{provider}:{backend}"
        return self.set(key, status, ttl)
    
    def get_cached_backend_status(self, provider: str, backend: str) -> Optional[Dict[str, Any]]:
        """Get cached backend status."""
        key = f"backend_status:{provider}:{backend}"
        return self.get(key)
    
    def cache_cost_calculation(self, experiment_hash: str, cost_data: Dict[str, Any], ttl: int = 1800):
        """Cache cost calculation results."""
        key = f"cost_calculation:{experiment_hash}"
        return self.set(key, cost_data, ttl)
    
    def get_cached_cost(self, experiment_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached cost calculation."""
        key = f"cost_calculation:{experiment_hash}"
        return self.get(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics if available."""
        if hasattr(self.backend, 'get_stats'):
            return self.backend.get_stats()
        
        # Basic stats
        keys = self.backend.keys()
        return {
            'total_keys': len(keys),
            'backend_type': type(self.backend).__name__
        }


# Global cache manager instance
_global_cache: Optional[CacheManager] = None
_cache_lock = threading.Lock()


def get_cache_manager(backend: Optional[CacheBackend] = None) -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = CacheManager(backend)
    
    return _global_cache


def configure_cache(cache_type: str = "memory", **kwargs) -> CacheManager:
    """Configure global cache with specified backend."""
    global _global_cache
    
    with _cache_lock:
        if cache_type == "memory":
            backend = MemoryCache(**kwargs)
        elif cache_type == "file":
            backend = FileCache(**kwargs)
        elif cache_type == "redis":
            backend = RedisCache(**kwargs)
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")
        
        _global_cache = CacheManager(backend)
    
    return _global_cache


# Convenience decorators using global cache
def cached(key: Optional[str] = None, ttl: Optional[int] = None):
    """Global cached decorator."""
    cache = get_cache_manager()
    return cache.cached(key, ttl)


def memoize(ttl: Optional[int] = None):
    """Global memoize decorator."""
    cache = get_cache_manager()
    return cache.memoize(ttl)