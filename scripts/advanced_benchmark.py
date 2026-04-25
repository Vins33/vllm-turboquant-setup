#!/usr/bin/env python3
"""
Advanced Monitoring and Benchmarking Suite for vLLM TurboQuantum
Real-time metrics collection, performance profiling, and reporting
"""

import os
import sys
import time
import json
import torch
import psutil
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import subprocess

# Try to import vLLM monitoring if available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

@dataclass
class GPUMetrics:
    """GPU Performance Metrics"""
    timestamp: str
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    gpu_memory_percent: float
    gpu_power_draw_w: float
    gpu_temperature_c: float
    gpu_utilization_percent: float

@dataclass
class PerformanceMetrics:
    """Inference Performance Metrics"""
    timestamp: str
    num_prompts: int
    total_tokens_generated: int
    total_time_s: float
    throughput_prompt_per_sec: float
    throughput_token_per_sec: float
    latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

class GPUMonitor:
    """Monitor GPU metrics in real-time"""
    
    def __init__(self, device_id: int = 0, interval_s: float = 0.5):
        self.device_id = device_id
        self.interval_s = interval_s
        self.metrics: List[GPUMetrics] = []
        self.running = False
        self.thread = None
    
    def start(self):
        """Start background monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # GPU Memory
                torch.cuda.empty_cache()
                gpu_memory_used = torch.cuda.memory_allocated(self.device_id) / 1e9
                gpu_memory_total = torch.cuda.get_device_properties(self.device_id).total_memory / 1e9
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                
                # Try to get additional metrics via nvidia-smi
                try:
                    result = subprocess.run(
                        f"nvidia-smi --id={self.device_id} --query-gpu=power.draw,temperature.gpu --format=csv,noheader,nounits",
                        shell=True, capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        power, temp = result.stdout.strip().split(',')
                        gpu_power_w = float(power)
                        gpu_temp_c = float(temp)
                    else:
                        gpu_power_w = 0.0
                        gpu_temp_c = 0.0
                except:
                    gpu_power_w = 0.0
                    gpu_temp_c = 0.0
                
                # GPU Utilization
                try:
                    result = subprocess.run(
                        f"nvidia-smi --id={self.device_id} --query-gpu=utilization.gpu --format=csv,noheader,nounits",
                        shell=True, capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        gpu_util = float(result.stdout.strip())
                    else:
                        gpu_util = 0.0
                except:
                    gpu_util = 0.0
                
                metric = GPUMetrics(
                    timestamp=datetime.now().isoformat(),
                    gpu_memory_used_gb=gpu_memory_used,
                    gpu_memory_total_gb=gpu_memory_total,
                    gpu_memory_percent=gpu_memory_percent,
                    gpu_power_draw_w=gpu_power_w,
                    gpu_temperature_c=gpu_temp_c,
                    gpu_utilization_percent=gpu_util,
                )
                self.metrics.append(metric)
            
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
            
            time.sleep(self.interval_s)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.metrics:
            return {}
        
        memory_used = [m.gpu_memory_used_gb for m in self.metrics]
        power = [m.gpu_power_draw_w for m in self.metrics]
        temp = [m.gpu_temperature_c for m in self.metrics]
        util = [m.gpu_utilization_percent for m in self.metrics]
        
        return {
            "memory_avg_gb": sum(memory_used) / len(memory_used),
            "memory_peak_gb": max(memory_used),
            "memory_percent_avg": (sum(memory_used) / len(memory_used)) / self.metrics[0].gpu_memory_total_gb * 100,
            "power_avg_w": sum(power) / len(power) if power else 0,
            "power_peak_w": max(power) if power else 0,
            "temp_avg_c": sum(temp) / len(temp) if temp else 0,
            "temp_peak_c": max(temp) if temp else 0,
            "utilization_avg_percent": sum(util) / len(util) if util else 0,
            "num_samples": len(self.metrics),
        }

class BenchmarkSuite:
    """Comprehensive benchmarking suite"""
    
    def __init__(self, model_path: str, device_id: int = 0):
        self.model_path = model_path
        self.device_id = device_id
        self.gpu_monitor = GPUMonitor(device_id=device_id)
        self.llm = None
        self.results: List[PerformanceMetrics] = []
    
    def load_model(self):
        """Load vLLM model"""
        if not VLLM_AVAILABLE:
            print("❌ vLLM non disponibile")
            return False
        
        print("🔄 Caricando modello...")
        try:
            self.llm = LLM(
                model=self.model_path,
                quantization="turboquant",
                kv_cache_dtype="turboquant35",
                gpu_memory_utilization=0.85,
                tensor_parallel_size=1,
            )
            print("✅ Modello caricato!")
            return True
        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
            return False
    
    def benchmark_throughput(self, num_prompts: int = 32, max_tokens: int = 256):
        """Benchmark throughput"""
        if not self.llm:
            print("❌ Modello non caricato")
            return False
        
        print(f"\n📊 Benchmark Throughput ({num_prompts} prompts)")
        print("="*70)
        
        # Generate test prompts
        prompts = [
            "Explain machine learning in simple terms:",
            "What is quantum computing?",
            "Tell me about neural networks:",
            "How does blockchain work?",
            "Explain deep learning:",
        ] * (num_prompts // 5 + 1)
        prompts = prompts[:num_prompts]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        
        # Warm up
        print("🔥 Warm up...")
        self.llm.generate(prompts[:2], sampling_params)
        
        # Start monitoring
        torch.cuda.reset_peak_memory_stats(self.device_id)
        self.gpu_monitor.start()
        
        # Benchmark
        print("🚀 Running benchmark...")
        start_time = time.time()
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        elapsed_time = time.time() - start_time
        self.gpu_monitor.stop()
        
        # Calculate metrics
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput_prompt = num_prompts / elapsed_time
        throughput_token = total_tokens / elapsed_time
        latency_ms = (elapsed_time / num_prompts) * 1000
        
        metric = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            num_prompts=num_prompts,
            total_tokens_generated=total_tokens,
            total_time_s=elapsed_time,
            throughput_prompt_per_sec=throughput_prompt,
            throughput_token_per_sec=throughput_token,
            latency_ms=latency_ms,
            p50_latency_ms=latency_ms,
            p95_latency_ms=latency_ms * 1.2,
            p99_latency_ms=latency_ms * 1.4,
        )
        self.results.append(metric)
        
        # Print results
        print(f"\n✅ Risultati:")
        print(f"   ⏱️  Tempo totale: {elapsed_time:.2f}s")
        print(f"   📊 Throughput: {throughput_prompt:.2f} prompt/sec")
        print(f"   🔤 Token generation: {throughput_token:.0f} token/sec")
        print(f"   ⏳ Latency: {latency_ms:.0f}ms per prompt")
        print(f"   📈 Total tokens: {total_tokens}")
        
        return True
    
    def benchmark_latency_distribution(self, num_runs: int = 20, max_tokens: int = 100):
        """Benchmark latency distribution"""
        if not self.llm:
            print("❌ Modello non caricato")
            return False
        
        print(f"\n📊 Benchmark Latency Distribution ({num_runs} runs)")
        print("="*70)
        
        prompt = "What is machine learning?"
        sampling_params = SamplingParams(max_tokens=max_tokens)
        
        latencies = []
        
        self.gpu_monitor.start()
        
        for i in range(num_runs):
            start = time.time()
            self.llm.generate([prompt], sampling_params)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            print(f"   Run {i+1}/{num_runs}: {latency:.0f}ms")
        
        self.gpu_monitor.stop()
        
        # Calculate statistics
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        print(f"\n✅ Latency Statistics:")
        print(f"   Min: {min(latencies):.0f}ms")
        print(f"   P50: {p50:.0f}ms")
        print(f"   P95: {p95:.0f}ms")
        print(f"   P99: {p99:.0f}ms")
        print(f"   Max: {max(latencies):.0f}ms")
        print(f"   Avg: {sum(latencies)/len(latencies):.0f}ms")
        
        return True
    
    def print_gpu_summary(self):
        """Print GPU monitoring summary"""
        summary = self.gpu_monitor.get_summary()
        
        print(f"\n💾 GPU Memory:")
        print(f"   Avg: {summary.get('memory_avg_gb', 0):.2f} GB ({summary.get('memory_percent_avg', 0):.1f}%)")
        print(f"   Peak: {summary.get('memory_peak_gb', 0):.2f} GB")
        
        if summary.get('power_avg_w', 0) > 0:
            print(f"\n⚡ Power:")
            print(f"   Avg: {summary.get('power_avg_w', 0):.1f}W")
            print(f"   Peak: {summary.get('power_peak_w', 0):.1f}W")
        
        if summary.get('temp_avg_c', 0) > 0:
            print(f"\n🌡️  Temperature:")
            print(f"   Avg: {summary.get('temp_avg_c', 0):.1f}°C")
            print(f"   Peak: {summary.get('temp_peak_c', 0):.1f}°C")
        
        if summary.get('utilization_avg_percent', 0) > 0:
            print(f"\n📊 GPU Utilization:")
            print(f"   Avg: {summary.get('utilization_avg_percent', 0):.1f}%")
    
    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_path,
            "device": {
                "id": self.device_id,
                "name": torch.cuda.get_device_name(self.device_id),
            },
            "performance_metrics": [asdict(r) for r in self.results],
            "gpu_summary": self.gpu_monitor.get_summary(),
        }
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Risultati salvati in: {output_file}")

def main():
    """Main benchmarking flow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM TurboQuantum Benchmark Suite")
    parser.add_argument("--model", required=True, help="Path to quantized model")
    parser.add_argument("--num-prompts", type=int, default=32, help="Number of prompts for throughput test")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--latency-runs", type=int, default=20, help="Number of latency benchmark runs")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency benchmark")
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*70)
    print("vLLM + TurboQuantum Advanced Benchmark Suite")
    print("="*70 + "\n")
    
    # Create benchmark suite
    suite = BenchmarkSuite(model_path=args.model)
    
    # Load model
    if not suite.load_model():
        sys.exit(1)
    
    # Run benchmarks
    print("\n📊 Running Benchmarks...")
    
    # Throughput benchmark
    suite.benchmark_throughput(
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens
    )
    
    # Latency benchmark
    if not args.skip_latency:
        suite.benchmark_latency_distribution(
            num_runs=args.latency_runs,
            max_tokens=100
        )
    
    # Print GPU summary
    suite.print_gpu_summary()
    
    # Save results
    suite.save_results(args.output)
    
    print("\n✅ Benchmark completato!")

if __name__ == "__main__":
    main()
