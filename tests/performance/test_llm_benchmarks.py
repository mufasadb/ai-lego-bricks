"""
Performance benchmarks for LLM services.
"""

import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

from llm.generation_service import GenerationService, create_generation_service
from llm.llm_types import GenerationConfig


class TestLLMPerformance:
    """Performance benchmarks for LLM operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_prompts = [
            "Hello, how are you?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot learning to paint.",
            "Analyze this data and provide insights: [1, 2, 3, 4, 5]",
            "What are the benefits of renewable energy?",
            "Debug this Python code: print('hello world')",
            "Translate 'Hello world' to Spanish, French, and German.",
            "Create a marketing plan for a new product.",
            "Explain the concept of machine learning to a 10-year-old.",
            "Generate a recipe for chocolate chip cookies."
        ]
        
        self.long_prompt = "Please analyze this lengthy document: " + "This is a sample sentence. " * 500
    
    @pytest.mark.performance
    def test_single_generation_performance(self, mock_environment_variables):
        """Benchmark single text generation."""
        with patch('llm.generation_service.LLMFactory') as mock_factory_class:
            # Mock response time proportional to prompt length
            def mock_generate(prompt, **kwargs):
                word_count = len(prompt.split())
                time.sleep(0.002 * word_count)  # 2ms per word
                return f"Generated response for prompt with {word_count} words"
            
            mock_client = Mock()
            mock_client.generate.side_effect = mock_generate
            
            mock_factory = Mock()
            mock_factory.create_client.return_value = mock_client
            mock_factory_class.return_value = mock_factory
            
            service = create_generation_service("auto")
            
            # Warm up
            service.generate("warmup prompt")
            
            # Benchmark short prompts
            execution_times = []
            for prompt in self.test_prompts:
                start_time = time.time()
                response = service.generate(prompt)
                end_time = time.time()
                execution_times.append(end_time - start_time)
                assert response is not None
            
            # Analyze results
            avg_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            p95_time = statistics.quantiles(execution_times, n=20)[18]
            
            print(f"\nLLM Generation Performance (short prompts):")
            print(f"Average time: {avg_time:.4f}s")
            print(f"Median time: {median_time:.4f}s")
            print(f"95th percentile: {p95_time:.4f}s")
            
            # Performance assertions
            assert avg_time < 0.5, f"Average generation time {avg_time:.4f}s exceeds 500ms threshold"
            assert p95_time < 1.0, f"95th percentile {p95_time:.4f}s exceeds 1s threshold"
    
    @pytest.mark.performance
    def test_streaming_generation_performance(self, mock_environment_variables):
        """Benchmark streaming text generation."""
        with patch('llm.generation_service.LLMFactory') as mock_factory_class:
            def mock_generate_stream(prompt, **kwargs):
                word_count = len(prompt.split())
                # Simulate streaming with chunks
                for i in range(min(word_count, 20)):  # Max 20 chunks
                    time.sleep(0.01)  # 10ms per chunk
                    yield f"chunk_{i} "
            
            mock_client = Mock()
            mock_client.generate_stream.side_effect = mock_generate_stream
            
            mock_factory = Mock()
            mock_factory.create_client.return_value = mock_client
            mock_factory_class.return_value = mock_factory
            
            service = create_generation_service("auto")
            
            # Test streaming performance
            execution_times = []
            chunk_counts = []
            
            for prompt in self.test_prompts[:5]:  # Test subset for streaming
                start_time = time.time()
                chunks = list(service.generate_stream(prompt))
                end_time = time.time()
                
                execution_times.append(end_time - start_time)
                chunk_counts.append(len(chunks))
                assert len(chunks) > 0
            
            avg_time = statistics.mean(execution_times)
            avg_chunks = statistics.mean(chunk_counts)
            time_per_chunk = avg_time / avg_chunks if avg_chunks > 0 else 0
            
            print(f"\nLLM Streaming Performance:")
            print(f"Average total time: {avg_time:.4f}s")
            print(f"Average chunks: {avg_chunks:.1f}")
            print(f"Average time per chunk: {time_per_chunk:.4f}s")
            
            # Performance assertions
            assert avg_time < 1.0, "Streaming taking too long overall"
            assert time_per_chunk < 0.05, "Individual chunks taking too long"
    
    @pytest.mark.performance
    def test_batch_generation_performance(self, mock_environment_variables):
        """Benchmark batch text generation."""
        with patch('llm.generation_service.LLMFactory') as mock_factory_class:
            def mock_generate(prompt, **kwargs):
                word_count = len(prompt.split())
                time.sleep(0.001 * word_count)  # Faster for batch processing
                return f"Batch response for: {prompt[:20]}..."
            
            mock_client = Mock()
            mock_client.generate.side_effect = mock_generate
            
            mock_factory = Mock()
            mock_factory.create_client.return_value = mock_client
            mock_factory_class.return_value = mock_factory
            
            service = create_generation_service("auto")
            
            # Test batch processing
            batch_sizes = [5, 10, 20, 50]
            
            for batch_size in batch_sizes:
                batch_prompts = self.test_prompts[:batch_size] if batch_size <= len(self.test_prompts) else \
                               (self.test_prompts * (batch_size // len(self.test_prompts) + 1))[:batch_size]
                
                start_time = time.time()
                responses = []
                for prompt in batch_prompts:
                    response = service.generate(prompt)
                    responses.append(response)
                end_time = time.time()
                
                total_time = end_time - start_time
                time_per_prompt = total_time / batch_size
                throughput = batch_size / total_time
                
                print(f"Batch size {batch_size}: {total_time:.4f}s total, {time_per_prompt:.4f}s per prompt, {throughput:.1f} prompts/s")
                
                assert len(responses) == batch_size
                assert time_per_prompt < 0.1, f"Batch processing too slow: {time_per_prompt:.4f}s per prompt"
    
    @pytest.mark.performance
    def test_concurrent_generation_performance(self, mock_environment_variables):
        """Benchmark concurrent text generation."""
        with patch('llm.generation_service.LLMFactory') as mock_factory_class:
            def mock_generate(prompt, **kwargs):
                word_count = len(prompt.split())
                time.sleep(0.002 * word_count)  # Simulate API call
                return f"Concurrent response for: {prompt[:30]}..."
            
            mock_client = Mock()
            mock_client.generate.side_effect = mock_generate
            
            mock_factory = Mock()
            mock_factory.create_client.return_value = mock_client
            mock_factory_class.return_value = mock_factory
            
            service = create_generation_service("auto")
            
            def generate_task(prompt_index):
                prompt = self.test_prompts[prompt_index % len(self.test_prompts)]
                start_time = time.time()
                response = service.generate(f"{prompt} (task {prompt_index})")
                end_time = time.time()
                return {
                    "index": prompt_index,
                    "time": end_time - start_time,
                    "response": response
                }
            
            # Test with different concurrency levels
            concurrency_levels = [1, 2, 5, 10]
            
            for max_workers in concurrency_levels:
                num_tasks = 20
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    start_time = time.time()
                    futures = [executor.submit(generate_task, i) for i in range(num_tasks)]
                    results = [future.result() for future in as_completed(futures)]
                    total_time = time.time() - start_time
                
                individual_times = [result["time"] for result in results]
                avg_individual_time = statistics.mean(individual_times)
                throughput = num_tasks / total_time
                
                print(f"Concurrency {max_workers}: {total_time:.4f}s total, {avg_individual_time:.4f}s avg individual, {throughput:.1f} req/s")
                
                assert len(results) == num_tasks
                assert total_time < 30.0, f"Concurrent processing took too long: {total_time:.4f}s"
    
    @pytest.mark.performance
    def test_prompt_length_scaling(self, mock_environment_variables):
        """Test performance scaling with prompt length."""
        with patch('llm.generation_service.LLMFactory') as mock_factory_class:
            def mock_generate(prompt, **kwargs):
                word_count = len(prompt.split())
                # Simulate realistic scaling: time increases with prompt length
                base_time = 0.1
                scaling_factor = 0.001
                time.sleep(base_time + (scaling_factor * word_count))
                return f"Response to {word_count}-word prompt"
            
            mock_client = Mock()
            mock_client.generate.side_effect = mock_generate
            
            mock_factory = Mock()
            mock_factory.create_client.return_value = mock_client
            mock_factory_class.return_value = mock_factory
            
            service = create_generation_service("auto")
            
            # Test different prompt lengths
            base_prompt = "Analyze this: "
            prompt_lengths = [10, 50, 100, 500, 1000, 2000]  # words
            
            performance_data = {}
            
            for length in prompt_lengths:
                # Create prompt of specified length
                content = "word " * length
                prompt = base_prompt + content
                
                execution_times = []
                for i in range(5):  # Run each test 5 times
                    start_time = time.time()
                    response = service.generate(prompt)
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                    assert response is not None
                
                avg_time = statistics.mean(execution_times)
                words_per_second = length / avg_time
                performance_data[length] = {
                    "avg_time": avg_time,
                    "words_per_second": words_per_second
                }
                
                print(f"Length {length} words: {avg_time:.4f}s ({words_per_second:.1f} words/s)")
            
            # Verify scaling behavior
            for i in range(1, len(prompt_lengths)):
                current_length = prompt_lengths[i]
                previous_length = prompt_lengths[i-1]
                
                current_time = performance_data[current_length]["avg_time"]
                previous_time = performance_data[previous_length]["avg_time"]
                
                # Time should scale reasonably with length
                time_ratio = current_time / previous_time
                length_ratio = current_length / previous_length
                
                # Allow some overhead but shouldn't be worse than linear
                assert time_ratio < length_ratio * 1.5, f"Poor scaling at length {current_length}"
    
    @pytest.mark.performance
    def test_generation_config_impact(self, mock_environment_variables):
        """Test performance impact of different generation configurations."""
        with patch('llm.generation_service.LLMFactory') as mock_factory_class:
            def mock_generate(prompt, config=None, **kwargs):
                base_time = 0.05
                
                # Simulate config impact on performance
                if config:
                    if config.max_tokens > 1000:
                        base_time *= 2  # Longer responses take more time
                    if config.temperature > 0.8:
                        base_time *= 1.5  # Higher creativity takes more time
                
                time.sleep(base_time)
                return f"Response with config: {config}"
            
            mock_client = Mock()
            mock_client.generate.side_effect = mock_generate
            
            mock_factory = Mock()
            mock_factory.create_client.return_value = mock_client
            mock_factory_class.return_value = mock_factory
            
            service = create_generation_service("auto")
            
            # Test different configurations
            configs = {
                "default": None,
                "short_fast": GenerationConfig(max_tokens=100, temperature=0.3),
                "long_creative": GenerationConfig(max_tokens=2000, temperature=0.9),
                "balanced": GenerationConfig(max_tokens=500, temperature=0.7),
            }
            
            prompt = "Write a creative story about space exploration."
            
            for config_name, config in configs.items():
                execution_times = []
                
                for i in range(10):
                    start_time = time.time()
                    response = service.generate(prompt, config=config)
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                    assert response is not None
                
                avg_time = statistics.mean(execution_times)
                print(f"Config '{config_name}': {avg_time:.4f}s")
                
                # Performance assertions based on config
                if config_name == "short_fast":
                    assert avg_time < 0.1, f"Short fast config too slow: {avg_time:.4f}s"
                elif config_name == "long_creative":
                    assert avg_time < 0.2, f"Long creative config too slow: {avg_time:.4f}s"
    
    @pytest.mark.performance
    def test_provider_switching_performance(self, mock_environment_variables):
        """Test performance of switching between providers."""
        providers = ["anthropic", "openai", "google"]
        
        def create_mock_service(provider):
            def mock_generate(prompt, **kwargs):
                # Simulate different provider speeds
                provider_delays = {
                    "anthropic": 0.05,
                    "openai": 0.07,
                    "google": 0.06
                }
                time.sleep(provider_delays.get(provider, 0.05))
                return f"{provider} response to: {prompt[:20]}..."
            
            mock_client = Mock()
            mock_client.generate.side_effect = mock_generate
            return mock_client
        
        with patch('llm.generation_service.LLMFactory') as mock_factory_class:
            def mock_create_client(provider_enum):
                provider_name = str(provider_enum).split('.')[-1].lower()
                return create_mock_service(provider_name)
            
            mock_factory = Mock()
            mock_factory.create_client.side_effect = mock_create_client
            mock_factory_class.return_value = mock_factory
            
            # Test switching between providers
            switch_times = []
            generation_times = {}
            
            for provider in providers:
                # Measure service creation time
                start_time = time.time()
                service = create_generation_service(provider)
                switch_time = time.time() - start_time
                switch_times.append(switch_time)
                
                # Measure generation time
                start_time = time.time()
                response = service.generate("Test prompt for provider switching")
                generation_time = time.time() - start_time
                generation_times[provider] = generation_time
                
                print(f"Provider {provider}: switch {switch_time:.4f}s, generation {generation_time:.4f}s")
                
                assert response is not None
                assert switch_time < 0.1, f"Provider switching too slow for {provider}"
            
            avg_switch_time = statistics.mean(switch_times)
            print(f"Average provider switch time: {avg_switch_time:.4f}s")
            
            # Verify reasonable switching performance
            assert avg_switch_time < 0.05, f"Average provider switching too slow: {avg_switch_time:.4f}s"
    
    @pytest.mark.performance
    def test_error_recovery_performance(self, mock_environment_variables):
        """Test performance of error handling and recovery."""
        with patch('llm.generation_service.LLMFactory') as mock_factory_class:
            call_count = 0
            
            def mock_generate_with_errors(prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                
                # Fail first few attempts, then succeed
                if call_count % 3 == 1:
                    time.sleep(0.01)  # Quick failure
                    raise Exception("API rate limit exceeded")
                elif call_count % 3 == 2:
                    time.sleep(0.02)  # Slightly slower failure
                    raise Exception("Service temporarily unavailable")
                else:
                    time.sleep(0.05)  # Normal success time
                    return f"Success response after retries: {prompt[:20]}..."
            
            mock_client = Mock()
            mock_client.generate.side_effect = mock_generate_with_errors
            
            mock_factory = Mock()
            mock_factory.create_client.return_value = mock_client
            mock_factory_class.return_value = mock_factory
            
            # Mock retry mechanism
            def generate_with_retry(service, prompt, max_retries=3):
                for attempt in range(max_retries):
                    try:
                        return service.generate(prompt)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                        continue
            
            service = create_generation_service("auto")
            
            # Test error recovery performance
            recovery_times = []
            success_count = 0
            
            for i in range(10):
                start_time = time.time()
                try:
                    response = generate_with_retry(service, f"Test prompt {i}")
                    if response:
                        success_count += 1
                except Exception:
                    pass  # Count as failure
                end_time = time.time()
                
                recovery_times.append(end_time - start_time)
            
            avg_recovery_time = statistics.mean(recovery_times)
            success_rate = success_count / 10
            
            print(f"Error Recovery Performance:")
            print(f"Average recovery time: {avg_recovery_time:.4f}s")
            print(f"Success rate: {success_rate:.1%}")
            
            # Performance assertions
            assert avg_recovery_time < 1.0, f"Error recovery too slow: {avg_recovery_time:.4f}s"
            assert success_rate >= 0.8, f"Success rate too low: {success_rate:.1%}"
    
    @pytest.mark.performance
    def test_memory_usage_during_generation(self, mock_environment_variables):
        """Test memory usage patterns during text generation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        with patch('llm.generation_service.LLMFactory') as mock_factory_class:
            def mock_generate(prompt, **kwargs):
                # Simulate memory-intensive generation
                temp_data = ["x"] * 10000  # Create some temporary data
                time.sleep(0.01)
                return f"Response with memory usage for: {prompt[:20]}..."
            
            mock_client = Mock()
            mock_client.generate.side_effect = mock_generate
            
            mock_factory = Mock()
            mock_factory.create_client.return_value = mock_client
            mock_factory_class.return_value = mock_factory
            
            service = create_generation_service("auto")
            
            # Measure memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = initial_memory
            
            memory_samples = []
            
            for i in range(20):
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                peak_memory = max(peak_memory, current_memory)
                
                response = service.generate(f"Memory test prompt {i}")
                assert response is not None
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            peak_increase = peak_memory - initial_memory
            
            print(f"Memory Usage During Generation:")
            print(f"Initial: {initial_memory:.1f} MB")
            print(f"Final: {final_memory:.1f} MB")
            print(f"Peak: {peak_memory:.1f} MB")
            print(f"Total increase: {memory_increase:.1f} MB")
            print(f"Peak increase: {peak_increase:.1f} MB")
            
            # Memory usage assertions
            assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f} MB"
            assert peak_increase < 150, f"Peak memory increase too high: {peak_increase:.1f} MB"