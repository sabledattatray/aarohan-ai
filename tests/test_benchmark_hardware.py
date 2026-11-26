from aarohan.model import AAROHANConfig
from scripts.benchmark_hardware import BENCHMARK_CONFIGS, benchmark_one


def test_architecture_benchmark_config_is_in_aarohan_001_parameter_range() -> None:
    config = BENCHMARK_CONFIGS["aarohan_0_01_candidate"]
    # Count without allocating the full model: tied embedding/head plus 16*d^2 per block.
    estimated = config.vocab_size * config.width + config.layers * 16 * config.width**2 + config.width
    assert 5_000_000 <= estimated <= 10_000_000


def test_cpu_benchmark_smoke_reports_finite_throughput() -> None:
    result = benchmark_one(AAROHANConfig(vocab_size=32, context_length=8, layers=1, heads=2, width=16),
                          device_name="cpu", batch_size=1, warmup_steps=0, measured_steps=1)
    assert result["status"] == "completed"
    assert result["training_tokens_per_second"] > 0

