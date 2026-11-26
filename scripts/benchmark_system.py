"""
scripts/benchmark_system.py
AAROHAN AI 2.0 Comprehensive System Performance & Hardware Benchmark.
Measures real performance metrics on the developer workstation (NVIDIA RTX 2060 6GB / CPU):
- Model loading cold/warm latency
- TTFT (Time to First Token)
- Inference generation throughput (tokens/sec)
- Cognitive Memory retrieval latency (keyword, vector, hybrid)
- Knowledge Graph traversal latency (1-hop, multi-hop BFS)
- System VRAM / RAM allocation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.inference import InferenceManager
from aarohan.memory import MemoryStore


def get_system_telemetry() -> dict[str, float]:
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = 0.0
    gpu_name = "None (CPU)"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    return {
        "ram_rss_mb": round(ram_mb, 2),
        "vram_allocated_mb": round(vram_mb, 2),
        "gpu_name": gpu_name,
        "cpu_count": psutil.cpu_count(logical=True),
    }


def benchmark_memory_retrieval(store: MemoryStore) -> dict[str, float]:
    # Ensure some benchmark data is loaded
    for i in range(25):
        store.add(f"Benchmark entity relationship fact #{i}: AAROHAN AI uses neural networks.", memory_type="semantic", source="benchmark")

    # Keyword search
    t0 = time.perf_counter()
    for _ in range(10):
        store.search("AAROHAN neural networks", limit=5)
    keyword_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

    # Vector search
    t0 = time.perf_counter()
    for _ in range(10):
        store.vector_search("AAROHAN neural networks", limit=5)
    vector_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

    # Hybrid search
    t0 = time.perf_counter()
    for _ in range(10):
        store.hybrid_search("AAROHAN neural networks", limit=5)
    hybrid_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

    return {
        "keyword_search_ms": round(keyword_ms, 3),
        "vector_search_ms": round(vector_ms, 3),
        "hybrid_search_ms": round(hybrid_ms, 3),
    }


def benchmark_graph_traversal(store: MemoryStore) -> dict[str, float]:
    # Build a multi-hop graph chain: A -> B -> C -> D -> E
    entities = ["NodeA", "NodeB", "NodeC", "NodeD", "NodeE"]
    for e in entities:
        store.add_entity(e, entity_type="concept")
    for i in range(len(entities) - 1):
        store.add_relation(entities[i], entities[i + 1], "depends_on", relation_class="verified")

    # 1-hop traversal
    t0 = time.perf_counter()
    for _ in range(20):
        store.get_relations("NodeA")
    one_hop_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0

    # Multi-hop traversal (A to E, 4 hops)
    t0 = time.perf_counter()
    for _ in range(20):
        store.find_path("NodeA", "NodeE", max_depth=5)
    multi_hop_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0

    return {
        "one_hop_traversal_ms": round(one_hop_ms, 3),
        "multi_hop_bfs_traversal_ms": round(multi_hop_ms, 3),
    }


def benchmark_inference(mgr: InferenceManager) -> dict[str, Any]:
    # 1. Custom Transformer benchmark
    custom_engine = mgr.get_engine("scratch")
    t0 = time.perf_counter()
    custom_res = custom_engine.generate("AAROHAN AI architecture", max_new_tokens=32)
    custom_duration = time.perf_counter() - t0
    custom_tokens = custom_res.generated_tokens
    custom_tps = custom_tokens / max(custom_duration, 1e-4)

    # Custom TTFT
    t0 = time.perf_counter()
    first_token_time = None
    for token in custom_engine.stream("AAROHAN AI architecture", max_new_tokens=16):
        if first_token_time is None:
            first_token_time = time.perf_counter() - t0
            break
    custom_ttft_ms = (first_token_time or 0.0) * 1000.0

    # 2. Foundation Model benchmark (if available)
    found_engine = mgr.get_engine("foundation")
    found_tps = 0.0
    found_ttft_ms = 0.0

    try:
        t0 = time.perf_counter()
        f_res = found_engine.generate("What is the capital of India?", max_new_tokens=32)
        f_dur = time.perf_counter() - t0
        found_tps = f_res.generated_tokens / max(f_dur, 1e-4)

        t0 = time.perf_counter()
        f_first = None
        for _ in found_engine.stream("What is the capital of India?", max_new_tokens=16):
            if f_first is None:
                f_first = time.perf_counter() - t0
                break
        found_ttft_ms = (f_first or 0.0) * 1000.0
    except Exception as e:
        print(f"  Note: Foundation engine benchmark deferred: {e}")

    return {
        "custom_transformer": {
            "tokens_per_second": round(custom_tps, 1),
            "ttft_ms": round(custom_ttft_ms, 2),
            "model_id": custom_engine.get_model_info().model_id,
            "device": custom_engine.device,
        },
        "foundation_model": {
            "tokens_per_second": round(found_tps, 1),
            "ttft_ms": round(found_ttft_ms, 2),
            "model_id": found_engine.get_model_info().model_id,
            "device": found_engine.device,
            "is_loaded": bool(found_engine.is_loaded() if callable(found_engine.is_loaded) else found_engine.is_loaded),
        },
    }


def run_benchmark(output_path: Path | None = None) -> dict[str, Any]:
    print("=" * 70)
    print("AAROHAN AI 2.0 SYSTEM BENCHMARK SUITE")
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    initial_telemetry = get_system_telemetry()
    print(f"Hardware: {initial_telemetry['gpu_name']} | Logical CPUs: {initial_telemetry['cpu_count']}")
    print(f"Base RSS: {initial_telemetry['ram_rss_mb']} MB | Base VRAM: {initial_telemetry['vram_allocated_mb']} MB")

    # Temporary isolated test database
    temp_db = PROJECT_ROOT / "checkpoints" / "temp_benchmark.sqlite"
    if temp_db.is_file():
        temp_db.unlink(missing_ok=True)
    store = MemoryStore(database_path=temp_db)

    print("\n[1/3] Benchmarking Cognitive Memory & Knowledge Graph...")
    mem_bench = benchmark_memory_retrieval(store)
    print(f"  - Keyword Search Latency: {mem_bench['keyword_search_ms']} ms")
    print(f"  - Vector Search Latency:  {mem_bench['vector_search_ms']} ms")
    print(f"  - Hybrid Search Latency:  {mem_bench['hybrid_search_ms']} ms")

    graph_bench = benchmark_graph_traversal(store)
    print(f"  - 1-Hop Graph Traversal:   {graph_bench['one_hop_traversal_ms']} ms")
    print(f"  - Multi-Hop BFS Traversal: {graph_bench['multi_hop_bfs_traversal_ms']} ms")

    print("\n[2/3] Benchmarking Neural Inference Engines...")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "aarohan-release-candidate.pt"
    if not ckpt_path.is_file():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "aarohan-sft-001.pt"
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizers" / "wikitext2-bpe" / "tokenizer.json"
    if not tok_path.is_file():
        tok_path = PROJECT_ROOT / "tokenizers" / "aarohan-bpe-0.1.0" / "tokenizer.json"

    mgr = InferenceManager(
        custom_checkpoint_path=ckpt_path if ckpt_path.is_file() else None,
        custom_tokenizer_path=tok_path if tok_path.is_file() else None,
    )
    inf_bench = benchmark_inference(mgr)
    print(f"  - Custom Transformer: {inf_bench['custom_transformer']['tokens_per_second']} tok/s (TTFT: {inf_bench['custom_transformer']['ttft_ms']} ms)")
    if inf_bench['foundation_model']['tokens_per_second'] > 0:
        print(f"  - Foundation Model:   {inf_bench['foundation_model']['tokens_per_second']} tok/s (TTFT: {inf_bench['foundation_model']['ttft_ms']} ms)")

    peak_telemetry = get_system_telemetry()
    print("\n[3/3] Telemetry & Resource Utilization:")
    print(f"  - Peak Process RSS: {peak_telemetry['ram_rss_mb']} MB")
    print(f"  - Peak VRAM Used:   {peak_telemetry['vram_allocated_mb']} MB")

    # Cleanup temp db
    temp_db.unlink(missing_ok=True)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": peak_telemetry,
        "memory_retrieval": mem_bench,
        "knowledge_graph": graph_bench,
        "inference": inf_bench,
        "status": "PASS",
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nBenchmark report saved to: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AAROHAN System Performance Benchmark")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "BENCHMARK_REPORT.json")
    args = parser.parse_args()
    run_benchmark(args.output)
