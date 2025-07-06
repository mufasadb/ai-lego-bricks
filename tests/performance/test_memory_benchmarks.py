"""
Performance benchmarks for memory services.
"""

import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

from memory.memory_factory import create_memory_service
from tests.fixtures.sample_data import SampleDocuments, SampleMetadata


class TestMemoryPerformance:
    """Performance benchmarks for memory operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sample_texts = [
            SampleDocuments.TECHNICAL_SPEC,
            SampleDocuments.RESEARCH_PAPER,
            SampleDocuments.USER_MANUAL,
            "Short text for testing",
            "Medium length text for testing performance " * 10,
            "Long text for testing performance and scalability " * 100
        ]
        
        self.sample_metadata = SampleMetadata.DOCUMENT_METADATA
    
    @pytest.mark.performance
    def test_single_memory_store_performance(self, mock_environment_variables):
        """Benchmark single memory store operation."""
        with patch('memory.memory_factory.create_memory_service') as mock_factory:
            mock_service = Mock()
            mock_service.store_memory.return_value = "test_memory_id"
            mock_factory.return_value = mock_service
            
            service = create_memory_service("auto")
            
            # Warm up
            service.store_memory("warmup text", {"test": True})
            
            # Benchmark
            execution_times = []
            for i in range(100):
                start_time = time.time()
                service.store_memory(f"Test memory {i}", {"index": i})
                end_time = time.time()
                execution_times.append(end_time - start_time)
            
            # Analyze results
            avg_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            p95_time = statistics.quantiles(execution_times, n=20)[18]  # 95th percentile
            
            print(f"\nMemory Store Performance:")
            print(f"Average time: {avg_time:.4f}s")
            print(f"Median time: {median_time:.4f}s") 
            print(f"95th percentile: {p95_time:.4f}s")
            
            # Performance assertions
            assert avg_time < 0.1, f"Average store time {avg_time:.4f}s exceeds 100ms threshold"
            assert p95_time < 0.2, f"95th percentile {p95_time:.4f}s exceeds 200ms threshold"
    
    @pytest.mark.performance
    def test_single_memory_retrieve_performance(self, mock_environment_variables):
        """Benchmark single memory retrieve operation."""
        with patch('memory.memory_factory.create_memory_service') as mock_factory:
            # Create mock memories with varying similarities
            mock_memories = []
            for i in range(10):
                mock_memory = Mock()
                mock_memory.content = f"Mock memory content {i}"
                mock_memory.similarity = 0.9 - (i * 0.05)
                mock_memory.metadata = {"index": i}
                mock_memories.append(mock_memory)
            
            mock_service = Mock()
            mock_service.retrieve_memories.return_value = mock_memories
            mock_factory.return_value = mock_service
            
            service = create_memory_service("auto")
            
            # Warm up
            service.retrieve_memories("warmup query", limit=5)
            
            # Benchmark
            execution_times = []
            for i in range(100):
                start_time = time.time()
                service.retrieve_memories(f"Test query {i}", limit=5)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            
            # Analyze results
            avg_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            p95_time = statistics.quantiles(execution_times, n=20)[18]
            
            print(f"\nMemory Retrieve Performance:")
            print(f"Average time: {avg_time:.4f}s")
            print(f"Median time: {median_time:.4f}s")
            print(f"95th percentile: {p95_time:.4f}s")
            
            # Performance assertions
            assert avg_time < 0.1, f"Average retrieve time {avg_time:.4f}s exceeds 100ms threshold"
            assert p95_time < 0.2, f"95th percentile {p95_time:.4f}s exceeds 200ms threshold"
    
    @pytest.mark.performance
    def test_bulk_memory_operations_performance(self, mock_environment_variables):
        """Benchmark bulk memory operations."""
        with patch('memory.memory_factory.create_memory_service') as mock_factory:
            mock_service = Mock()
            
            # Mock bulk store
            def mock_bulk_store(contents, metadatas):
                return [f"memory_id_{i}" for i in range(len(contents))]
            
            # Mock bulk retrieve
            def mock_bulk_retrieve(queries, limit):
                results = []
                for query in queries:
                    mock_memories = [Mock(content=f"Result for {query}", similarity=0.9)]
                    results.append(mock_memories)
                return results
            
            mock_service.bulk_store_memories = mock_bulk_store
            mock_service.bulk_retrieve_memories = mock_bulk_retrieve
            mock_factory.return_value = mock_service
            
            service = create_memory_service("auto")
            
            # Test bulk store performance
            bulk_contents = [f"Bulk memory content {i}" for i in range(50)]
            bulk_metadata = [{"index": i, "bulk": True} for i in range(50)]
            
            start_time = time.time()
            memory_ids = service.bulk_store_memories(bulk_contents, bulk_metadata)
            bulk_store_time = time.time() - start_time
            
            assert len(memory_ids) == 50
            print(f"\nBulk Store Performance (50 items): {bulk_store_time:.4f}s")
            
            # Test bulk retrieve performance
            bulk_queries = [f"Query {i}" for i in range(20)]
            
            start_time = time.time()
            bulk_results = service.bulk_retrieve_memories(bulk_queries, limit=5)
            bulk_retrieve_time = time.time() - start_time
            
            assert len(bulk_results) == 20
            print(f"Bulk Retrieve Performance (20 queries): {bulk_retrieve_time:.4f}s")
            
            # Performance assertions
            assert bulk_store_time < 2.0, f"Bulk store time {bulk_store_time:.4f}s exceeds 2s threshold"
            assert bulk_retrieve_time < 1.0, f"Bulk retrieve time {bulk_retrieve_time:.4f}s exceeds 1s threshold"
    
    @pytest.mark.performance
    def test_concurrent_memory_operations(self, mock_environment_variables):
        """Benchmark concurrent memory operations."""
        with patch('memory.memory_factory.create_memory_service') as mock_factory:
            mock_service = Mock()
            mock_service.store_memory.return_value = "concurrent_memory_id"
            mock_service.retrieve_memories.return_value = [
                Mock(content="Concurrent result", similarity=0.9)
            ]
            mock_factory.return_value = mock_service
            
            service = create_memory_service("auto")
            
            def store_operation(index):
                start_time = time.time()
                service.store_memory(f"Concurrent memory {index}", {"index": index})
                return time.time() - start_time
            
            def retrieve_operation(index):
                start_time = time.time()
                service.retrieve_memories(f"Concurrent query {index}", limit=3)
                return time.time() - start_time
            
            # Test concurrent stores
            with ThreadPoolExecutor(max_workers=5) as executor:
                start_time = time.time()
                store_futures = [executor.submit(store_operation, i) for i in range(20)]
                store_times = [future.result() for future in as_completed(store_futures)]
                concurrent_store_time = time.time() - start_time
            
            print(f"\nConcurrent Store Performance (20 operations, 5 workers):")
            print(f"Total time: {concurrent_store_time:.4f}s")
            print(f"Average operation time: {statistics.mean(store_times):.4f}s")
            
            # Test concurrent retrieves
            with ThreadPoolExecutor(max_workers=5) as executor:
                start_time = time.time()
                retrieve_futures = [executor.submit(retrieve_operation, i) for i in range(20)]
                retrieve_times = [future.result() for future in as_completed(retrieve_futures)]
                concurrent_retrieve_time = time.time() - start_time
            
            print(f"Concurrent Retrieve Performance (20 operations, 5 workers):")
            print(f"Total time: {concurrent_retrieve_time:.4f}s")
            print(f"Average operation time: {statistics.mean(retrieve_times):.4f}s")
            
            # Performance assertions
            assert concurrent_store_time < 5.0, "Concurrent stores took too long"
            assert concurrent_retrieve_time < 5.0, "Concurrent retrieves took too long"
    
    @pytest.mark.performance
    def test_memory_search_scaling(self, mock_environment_variables):
        """Test memory search performance with different result set sizes."""
        with patch('memory.memory_factory.create_memory_service') as mock_factory:
            def mock_retrieve_with_limit(query, limit):
                # Simulate varying response times based on limit
                time.sleep(0.001 * limit)  # Small delay proportional to limit
                
                results = []
                for i in range(min(limit, 100)):  # Max 100 results
                    mock_memory = Mock()
                    mock_memory.content = f"Search result {i} for {query}"
                    mock_memory.similarity = 0.9 - (i * 0.005)
                    results.append(mock_memory)
                return results
            
            mock_service = Mock()
            mock_service.retrieve_memories.side_effect = mock_retrieve_with_limit
            mock_factory.return_value = mock_service
            
            service = create_memory_service("auto")
            
            # Test different result set sizes
            limits = [1, 5, 10, 25, 50, 100]
            performance_data = {}
            
            for limit in limits:
                execution_times = []
                for i in range(10):  # Run each test 10 times
                    start_time = time.time()
                    results = service.retrieve_memories(f"Scale test query {i}", limit=limit)
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                    assert len(results) == limit
                
                avg_time = statistics.mean(execution_times)
                performance_data[limit] = avg_time
                
                print(f"Limit {limit}: Average time {avg_time:.4f}s")
            
            # Verify performance scales reasonably
            for i in range(1, len(limits)):
                current_limit = limits[i]
                previous_limit = limits[i-1]
                
                current_time = performance_data[current_limit]
                previous_time = performance_data[previous_limit]
                
                # Time should not increase more than 10x when limit increases
                time_ratio = current_time / previous_time
                limit_ratio = current_limit / previous_limit
                
                assert time_ratio < limit_ratio * 2, f"Performance degradation too high at limit {current_limit}"
    
    @pytest.mark.performance
    def test_memory_size_impact(self, mock_environment_variables):
        """Test performance impact of different memory content sizes."""
        with patch('memory.memory_factory.create_memory_service') as mock_factory:
            mock_service = Mock()
            
            # Mock store with delay proportional to content size
            def mock_store_with_size_delay(content, metadata):
                size_kb = len(content) / 1024
                time.sleep(0.001 * size_kb)  # 1ms per KB
                return f"memory_id_{len(content)}"
            
            mock_service.store_memory.side_effect = mock_store_with_size_delay
            mock_factory.return_value = mock_service
            
            service = create_memory_service("auto")
            
            # Test different content sizes
            sizes = {
                "tiny": "Small text",  # ~10 bytes
                "small": "Small text content " * 10,  # ~200 bytes
                "medium": "Medium text content " * 100,  # ~2KB
                "large": "Large text content " * 1000,  # ~20KB
                "xlarge": "Extra large text content " * 5000  # ~100KB
            }
            
            performance_data = {}
            
            for size_name, content in sizes.items():
                execution_times = []
                content_size = len(content)
                
                for i in range(10):
                    start_time = time.time()
                    memory_id = service.store_memory(content, {"size": size_name})
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                    assert memory_id is not None
                
                avg_time = statistics.mean(execution_times)
                performance_data[size_name] = {
                    "size_bytes": content_size,
                    "avg_time": avg_time,
                    "throughput_kb_per_sec": (content_size / 1024) / avg_time
                }
                
                print(f"{size_name} ({content_size:,} bytes): {avg_time:.4f}s")
                print(f"  Throughput: {performance_data[size_name]['throughput_kb_per_sec']:.2f} KB/s")
            
            # Verify reasonable performance for different sizes
            assert performance_data["tiny"]["avg_time"] < 0.01, "Tiny content taking too long"
            assert performance_data["small"]["avg_time"] < 0.02, "Small content taking too long"
            assert performance_data["medium"]["avg_time"] < 0.1, "Medium content taking too long"
            assert performance_data["large"]["avg_time"] < 0.5, "Large content taking too long"
    
    @pytest.mark.performance
    def test_memory_embedding_performance(self, mock_environment_variables):
        """Test embedding generation performance."""
        with patch('memory.memory_factory.create_memory_service') as mock_factory:
            # Mock embedding generation with realistic delays
            def mock_generate_embedding(text):
                # Simulate embedding generation time based on text length
                word_count = len(text.split())
                time.sleep(0.001 * word_count)  # 1ms per word
                return [0.1] * 384  # Mock 384-dimensional embedding
            
            mock_service = Mock()
            mock_service.generate_embedding.side_effect = mock_generate_embedding
            mock_factory.return_value = mock_service
            
            service = create_memory_service("auto")
            
            # Test different text lengths
            test_texts = [
                "Short",  # 1 word
                "This is a medium length text",  # 7 words
                "This is a longer text that contains more words and should take longer to process",  # 15 words
                " ".join(["word"] * 50),  # 50 words
                " ".join(["word"] * 100)  # 100 words
            ]
            
            for text in test_texts:
                word_count = len(text.split())
                
                execution_times = []
                for i in range(10):
                    start_time = time.time()
                    embedding = service.generate_embedding(text)
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                    assert len(embedding) == 384
                
                avg_time = statistics.mean(execution_times)
                words_per_second = word_count / avg_time
                
                print(f"Embedding {word_count} words: {avg_time:.4f}s ({words_per_second:.1f} words/s)")
                
                # Performance assertions based on word count
                if word_count <= 10:
                    assert avg_time < 0.05, f"Small text embedding too slow: {avg_time:.4f}s"
                elif word_count <= 50:
                    assert avg_time < 0.2, f"Medium text embedding too slow: {avg_time:.4f}s"
                else:
                    assert avg_time < 0.5, f"Large text embedding too slow: {avg_time:.4f}s"
    
    @pytest.mark.performance
    def test_memory_cache_performance(self, mock_environment_variables):
        """Test memory caching performance."""
        with patch('memory.memory_factory.create_memory_service') as mock_factory:
            # Simulate cache with faster lookups for repeated queries
            cache = {}
            
            def mock_retrieve_with_cache(query, limit):
                if query in cache:
                    # Cache hit - instant return
                    return cache[query]
                else:
                    # Cache miss - simulate database lookup
                    time.sleep(0.05)  # 50ms for "database" lookup
                    results = [Mock(content=f"Result for {query}", similarity=0.9)]
                    cache[query] = results
                    return results
            
            mock_service = Mock()
            mock_service.retrieve_memories.side_effect = mock_retrieve_with_cache
            mock_factory.return_value = mock_service
            
            service = create_memory_service("auto")
            
            # Test cache performance
            query = "test cache query"
            
            # First call (cache miss)
            start_time = time.time()
            results1 = service.retrieve_memories(query, limit=5)
            first_call_time = time.time() - start_time
            
            # Second call (cache hit)
            start_time = time.time()
            results2 = service.retrieve_memories(query, limit=5)
            second_call_time = time.time() - start_time
            
            print(f"\nCache Performance:")
            print(f"First call (miss): {first_call_time:.4f}s")
            print(f"Second call (hit): {second_call_time:.4f}s")
            print(f"Speedup: {first_call_time / second_call_time:.1f}x")
            
            # Cache should provide significant speedup
            assert second_call_time < first_call_time / 10, "Cache not providing sufficient speedup"
            assert second_call_time < 0.01, "Cache lookup too slow"
            assert results1 == results2, "Cache returned different results"