# src/evaluation/__init__.py
from .benchmarks        import run_benchmark, aggregate_results, save_results
from .latency           import measure_latency, benchmark_model, get_model_size_mb
from .statistical_tests import pairwise_wilcoxon, kruskal_wallis_test, cohen_d

__all__ = [
    "run_benchmark", "aggregate_results", "save_results",
    "measure_latency", "benchmark_model", "get_model_size_mb",
    "pairwise_wilcoxon", "kruskal_wallis_test", "cohen_d",
]
