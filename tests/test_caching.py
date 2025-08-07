"""
Test suite for caching system.
"""

import pytest
import tempfile
import time
import threading
from pathlib import Path

from quantum_devops_ci.caching import (
    MemoryCache, DiskCache, MultiLevelCache, CacheManager,
    cached, create_circuit_hash
)


class TestMemoryCache:
    """Test in-memory cache functionality."""
    
    def test_memory_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = MemoryCache(max_size_mb=1, max_entries=10)
        
        # Put and get
        assert cache.put("key1", "value1") is True
        assert cache.get("key1") == "value1"
        
        # Get non-existent key
        assert cache.get("nonexistent") is None
        
        # Delete key
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False
    
    def test_memory_cache_ttl(self):
        """Test TTL functionality."""
        cache = MemoryCache(max_size_mb=1, default_ttl_seconds=1)
        
        # Put with TTL
        cache.put("ttl_key", "ttl_value", ttl_seconds=1)
        assert cache.get("ttl_key") == "ttl_value"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("ttl_key") is None
    
    def test_memory_cache_eviction(self):
        """Test LRU eviction."""
        cache = MemoryCache(max_size_mb=1, max_entries=2)
        
        # Fill cache to capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add third key, should evict key2
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # New key
    
    def test_memory_cache_stats(self):
        """Test cache statistics."""
        cache = MemoryCache(max_size_mb=1, max_entries=10)
        
        # Generate some hits and misses
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['entries'] == 1
    
    def test_memory_cache_thread_safety(self):
        """Test thread safety of memory cache."""
        cache = MemoryCache(max_size_mb=1, max_entries=100)
        
        def worker(start_idx):
            for i in range(start_idx, start_idx + 10):
                cache.put(f"key{i}", f"value{i}")
                retrieved = cache.get(f"key{i}")
                assert retrieved == f"value{i}"
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i * 10,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify some data is present
        assert cache.get_stats()['entries'] > 0


class TestDiskCache:
    """Test disk-based cache functionality."""
    
    def test_disk_cache_basic_operations(self):
        """Test basic disk cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(temp_dir, max_size_mb=1)
            
            # Put and get
            assert cache.put("disk_key1", {"data": "value1"}) is True
            result = cache.get("disk_key1")
            assert result == {"data": "value1"}
            
            # Get non-existent key
            assert cache.get("nonexistent") is None
            
            # Delete key
            assert cache.delete("disk_key1") is True
            assert cache.get("disk_key1") is None
    
    def test_disk_cache_persistence(self):
        """Test that disk cache persists across instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First cache instance
            cache1 = DiskCache(temp_dir, max_size_mb=1)
            cache1.put("persistent_key", {"data": "persistent_value"})
            
            # Second cache instance
            cache2 = DiskCache(temp_dir, max_size_mb=1)
            result = cache2.get("persistent_key")
            assert result == {"data": "persistent_value"}
    
    def test_disk_cache_ttl(self):
        """Test disk cache TTL functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(temp_dir, max_size_mb=1)
            
            # Put with short TTL
            cache.put("ttl_key", "ttl_value", ttl_seconds=1)
            assert cache.get("ttl_key") == "ttl_value"
            
            # Wait for expiration
            time.sleep(1.1)
            assert cache.get("ttl_key") is None
    
    def test_disk_cache_size_limit(self):
        """Test disk cache size limits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(temp_dir, max_size_mb=1)
            
            # Try to store very large object
            large_data = "x" * (2 * 1024 * 1024)  # 2MB string
            assert cache.put("large_key", large_data) is False
    
    def test_disk_cache_stats(self):
        """Test disk cache statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(temp_dir, max_size_mb=1)
            
            cache.put("key1", "value1")
            cache.put("key2", "value2")
            cache.get("key1")
            
            stats = cache.get_stats()
            assert stats['entries'] == 2
            assert stats['total_accesses'] >= 1
            assert temp_dir in stats['cache_dir']


class TestMultiLevelCache:
    """Test multi-level cache functionality."""
    
    def test_multilevel_cache_basic(self):
        """Test basic multi-level cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = MultiLevelCache(
                memory_cache_mb=1,
                disk_cache_mb=2,
                cache_dir=temp_dir
            )
            
            # Put and get - should be in both levels
            cache.put("ml_key", "ml_value")
            assert cache.get("ml_key") == "ml_value"
            
            # Verify it's in memory cache
            assert cache.memory_cache.get("ml_key") == "ml_value"
    
    def test_multilevel_cache_promotion(self):
        """Test cache promotion from disk to memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = MultiLevelCache(
                memory_cache_mb=1,
                disk_cache_mb=2,
                cache_dir=temp_dir
            )
            
            # Put data
            cache.put("promo_key", "promo_value")
            
            # Clear memory cache but keep disk
            cache.memory_cache.clear()
            assert cache.memory_cache.get("promo_key") is None
            
            # Get from multi-level cache - should promote to memory
            result = cache.get("promo_key")
            assert result == "promo_value"
            assert cache.memory_cache.get("promo_key") == "promo_value"
    
    def test_multilevel_cache_stats(self):
        """Test multi-level cache statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = MultiLevelCache(
                memory_cache_mb=1,
                disk_cache_mb=2,
                cache_dir=temp_dir
            )
            
            cache.put("stats_key1", "stats_value1")
            cache.put("stats_key2", "stats_value2")
            
            stats = cache.get_stats()
            assert 'memory' in stats
            assert 'disk' in stats
            assert stats['total_entries'] >= 2


class TestCacheManager:
    """Test cache manager functionality."""
    
    def test_cache_manager_circuit_cache(self):
        """Test circuit result caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'circuit_cache_dir': temp_dir,
                'circuit_memory_mb': 10,
                'circuit_disk_mb': 20
            }
            
            cache_manager = CacheManager(config)
            
            # Cache circuit result
            circuit_hash = "test_circuit_hash"
            backend = "qasm_simulator"
            shots = 1000
            result = {"counts": {"00": 500, "11": 500}}
            
            assert cache_manager.cache_circuit_result(circuit_hash, backend, shots, result) is True
            
            # Retrieve cached result
            cached_result = cache_manager.get_circuit_result(circuit_hash, backend, shots)
            assert cached_result == result
    
    def test_cache_manager_backend_info(self):
        """Test backend info caching."""
        cache_manager = CacheManager()
        
        backend_name = "test_backend"
        backend_info = {
            "name": backend_name,
            "qubits": 5,
            "topology": "linear"
        }
        
        # Cache and retrieve
        assert cache_manager.cache_backend_info(backend_name, backend_info) is True
        cached_info = cache_manager.get_backend_info(backend_name)
        assert cached_info == backend_info
    
    def test_cache_manager_cost_estimates(self):
        """Test cost estimate caching."""
        cache_manager = CacheManager()
        
        job_spec_hash = "job_hash_123"
        cost_estimate = {
            "total_cost": 15.50,
            "breakdown": {"simulator": 0.0, "hardware": 15.50}
        }
        
        # Cache and retrieve
        assert cache_manager.cache_cost_estimate(job_spec_hash, cost_estimate) is True
        cached_estimate = cache_manager.get_cost_estimate(job_spec_hash)
        assert cached_estimate == cost_estimate
    
    def test_cache_manager_global_stats(self):
        """Test global cache statistics."""
        cache_manager = CacheManager()
        
        # Add some cached data
        cache_manager.cache_circuit_result("hash1", "backend1", 1000, {"result": "data1"})
        cache_manager.cache_backend_info("backend1", {"info": "data"})
        cache_manager.cache_cost_estimate("hash1", {"cost": 10.0})
        
        stats = cache_manager.get_global_stats()
        assert 'circuit' in stats
        assert 'config' in stats
        assert 'backend' in stats
        assert 'cost' in stats


class TestCachedDecorator:
    """Test cached decorator functionality."""
    
    def test_cached_decorator(self):
        """Test cached decorator with circuit cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager({
                'circuit_cache_dir': temp_dir,
                'circuit_memory_mb': 10
            })
            
            call_count = 0
            
            @cached(cache_manager, 'circuit', ttl_seconds=300)
            def expensive_function(x, y):
                nonlocal call_count
                call_count += 1
                return x + y
            
            # First call
            result1 = expensive_function(5, 3)
            assert result1 == 8
            assert call_count == 1
            
            # Second call with same args - should use cache
            result2 = expensive_function(5, 3)
            assert result2 == 8
            assert call_count == 1  # Should not increment
            
            # Third call with different args
            result3 = expensive_function(10, 2)
            assert result3 == 12
            assert call_count == 2  # Should increment


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_circuit_hash(self):
        """Test circuit hash creation."""
        # Mock circuit object
        class MockCircuit:
            def __init__(self, num_qubits, depth_val):
                self.num_qubits = num_qubits
                self._depth = depth_val
            
            def depth(self):
                return self._depth
            
            def __str__(self):
                return f"MockCircuit({self.num_qubits}, {self._depth})"
        
        circuit1 = MockCircuit(5, 10)
        circuit2 = MockCircuit(5, 10)  # Same as circuit1
        circuit3 = MockCircuit(5, 15)  # Different depth
        
        hash1 = create_circuit_hash(circuit1)
        hash2 = create_circuit_hash(circuit2)
        hash3 = create_circuit_hash(circuit3)
        
        # Same circuits should have same hash
        assert hash1 == hash2
        # Different circuits should have different hash
        assert hash1 != hash3
        
        # Hash should be valid hex string
        assert len(hash1) == 32  # MD5 hex length
        int(hash1, 16)  # Should not raise exception


if __name__ == '__main__':
    pytest.main([__file__])