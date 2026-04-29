"""Main Auto-Tuner optimizer."""

import asyncio
import concurrent.futures
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .config import (
    SearchSpace,
    TuningConfig,
    TuningResult,
    Objective,
    get_default_vllm_space,
)
from .search import SearchStrategy, create_search_strategy
from .evaluator import ConfigEvaluator


def _run_async(coro):
    """Run async coroutine, handling existing event loops (Streamlit/uvloop compatible)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in an event loop (Streamlit/uvloop), run in a separate thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # No event loop, safe to use asyncio.run directly
        return asyncio.run(coro)


@dataclass
class TuningProgress:
    """Progress tracking for tuning session."""
    total_trials: int = 0
    completed_trials: int = 0
    best_score: float = float("-inf")
    best_config: Optional[TuningConfig] = None
    current_trial: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())


class AutoTuner:
    """Main Auto-Tuner for vLLM parameter optimization."""

    def __init__(
        self,
        model_path: str,
        gpu_ids: str,
        search_space: Optional[SearchSpace] = None,
        strategy: str = "bayesian",
        objective: str = "throughput",
        device: str = "nvidia",
        base_port: int = 8100,
        max_trials: int = 20,
        startup_timeout: int = 300,
        seed: int = 42,
        verbose: bool = True,
        log_dir: str = "./results/autotune/logs",
        progress_callback: Optional[Callable[[TuningProgress], None]] = None,
    ):
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.device = device
        self.search_space = search_space or get_default_vllm_space(
            gpu_count=len(gpu_ids.split(","))
        )
        self.strategy_name = strategy
        self.objective = Objective(objective)
        self.max_trials = max_trials
        self.seed = seed
        self.verbose = verbose
        self.log_dir = log_dir
        self.progress_callback = progress_callback

        # Initialize components
        self.search_strategy = create_search_strategy(
            strategy,
            n_startup_trials=min(5, max_trials // 4),
            seed=seed,
        )
        self.evaluator = ConfigEvaluator(
            model_path=model_path,
            gpu_ids=gpu_ids,
            device=device,
            base_port=base_port,
            log_dir=log_dir,
            startup_timeout=startup_timeout,
            verbose=verbose,
        )

        # State
        self.results: List[TuningResult] = []
        self.progress = TuningProgress(total_trials=max_trials)
        self._study = None

    def run(self) -> TuningResult:
        """Run the auto-tuning process (synchronous wrapper)."""
        return _run_async(self._run_optimization())

    async def _run_optimization(self) -> TuningResult:
        """Internal async implementation of optimization."""
        import optuna

        if self.verbose:
            print("\n" + "=" * 60)
            print("AUTO-TUNER: vLLM Parameter Optimization")
            print("=" * 60)
            print(f"Model: {self.model_path}")
            print(f"GPU IDs: {self.gpu_ids}")
            print(f"Device: {self.device}")
            print(f"Strategy: {self.strategy_name}")
            print(f"Objective: {self.objective.value}")
            print(f"Max Trials: {self.max_trials}")
            print(f"Search Space:")
            for param in self.search_space.parameters:
                if param.values:
                    print(f"  - {param.name}: {param.values}")
                else:
                    print(f"  - {param.name}: [{param.min_val}, {param.max_val}]")
            print("=" * 60)

        # Create study
        direction = "maximize"
        self._study = self.search_strategy.create_study(direction=direction)

        # Define objective function - must be sync for optuna
        def objective(trial: optuna.Trial) -> float:
            # Run async in a new thread to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._evaluate_trial(trial)
                )
                try:
                    # Add timeout to prevent hanging
                    return future.result(timeout=self.evaluator.startup_timeout + 600)
                except Exception as e:
                    if self.verbose:
                        print(f"[Trial {trial.number + 1}] Execution error: {e}")
                    return float("-inf")

        # Run optimization
        try:
            if self.strategy_name == "bayesian":
                # Use optuna's optimization loop for Bayesian
                self._study.optimize(
                    objective,
                    n_trials=self.max_trials,
                    catch=(Exception,),
                )
            else:
                # Manual loop for random/grid (sync, so no await)
                for trial_id in range(self.max_trials):
                    # Run each evaluation in a separate thread
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._evaluate_trial_manual(trial_id)
                        )
                        try:
                            result = future.result(timeout=self.evaluator.startup_timeout + 300)
                            self._record_result(result)
                        except Exception as e:
                            if self.verbose:
                                print(f"[Trial {trial_id + 1}] Error: {e}")

        except KeyboardInterrupt:
            if self.verbose:
                print("\n[AutoTuner] Interrupted by user")

        finally:
            # Cleanup in a new thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(asyncio.run, self.evaluator.cleanup()).result()

        # Get best result
        best_result = self._get_best_result()

        if self.verbose and best_result:
            self._print_summary(best_result)

        return best_result

    async def _evaluate_trial(self, trial) -> float:
        """Evaluate a single optuna trial."""
        trial_id = trial.number + 1
        self.progress.current_trial = trial_id

        try:
            config = self.search_strategy.suggest_config(trial, self.search_space)
            config.device = self.device  # Set device from AutoTuner
            result = await self.evaluator.evaluate(
                config, trial_id, self.objective.value
            )
            self._record_result(result)

            # Log the result
            if result.error:
                if self.verbose:
                    import traceback
                    print(f"[Trial {trial_id}] Failed: {result.error}")
            else:
                if self.verbose:
                    print(f"[Trial {trial_id}] Score: {result.score:.2f}, TPS: {result.tps:.2f}")

            return result.score
        except Exception as e:
            import traceback
            if self.verbose:
                print(f"[Trial {trial_id}] Exception: {e}")
                traceback.print_exc()
            # Record failed result
            failed_result = TuningResult(
                trial_id=trial_id,
                config=TuningConfig(),
                error=str(e),
            )
            failed_result.score = float("-inf")
            self.results.append(failed_result)
            return float("-inf")

    async def _evaluate_trial_manual(self, trial_id: int) -> TuningResult:
        """Evaluate a single trial for manual search strategies."""
        trial = {"trial_id": trial_id + 1}
        config = self.search_strategy.suggest_config(trial, self.search_space)
        config.device = self.device  # Set device from AutoTuner
        result = await self.evaluator.evaluate(
            config, trial_id + 1, self.objective.value
        )
        self.search_strategy.update(trial, result)
        return result

    def _record_result(self, result: TuningResult):
        """Record and process result."""
        self.results.append(result)
        self.progress.completed_trials += 1

        # Check constraints
        valid, violations = self.search_space.validate_result(result.metrics)
        result.constraint_violations = violations

        # Update best
        if result.score > self.progress.best_score:
            self.progress.best_score = result.score
            self.progress.best_config = result.config

        # Callback
        if self.progress_callback:
            self.progress_callback(self.progress)

        # Print progress
        if self.verbose:
            status = "✓" if valid else "✗ (constraints)"
            print(f"[Trial {result.trial_id}] {status}")
            print(f"  Score: {result.score:.2f}")
            print(f"  TPS: {result.tps:.2f} tokens/s")
            print(f"  P99: {result.latency_p99:.2f} ms")

    def _get_best_result(self) -> Optional[TuningResult]:
        """Get best result from all trials."""
        valid_results = [
            r for r in self.results
            if r.error is None and len(r.constraint_violations) == 0
        ]

        if not valid_results:
            # Fall back to all results
            valid_results = [r for r in self.results if r.error is None]

        if not valid_results:
            return None

        return max(valid_results, key=lambda r: r.score)

    def _print_summary(self, best: TuningResult):
        """Print tuning summary."""
        print("\n" + "=" * 60)
        print("AUTO-TUNING COMPLETED")
        print("=" * 60)
        print(f"Total Trials: {len(self.results)}")
        print(f"Successful: {sum(1 for r in self.results if r.error is None)}")
        print(f"Best Score: {best.score:.2f}")

        print("\n[Best Configuration]")
        print(f"  gpu_memory_utilization: {best.config.gpu_memory_utilization}")
        print(f"  tensor_parallel: {best.config.tensor_parallel}")
        print(f"  max_model_len: {best.config.max_model_len}")
        print(f"  max_num_seqs: {best.config.max_num_seqs}")

        print("\n[Best Metrics]")
        print(f"  TPS: {best.tps:.2f} tokens/s")
        print(f"  QPS: {best.metrics.get('qps', 0):.2f} req/s")
        print(f"  Latency P50: {best.metrics.get('latency_p50', 0):.2f} ms")
        print(f"  Latency P99: {best.latency_p99:.2f} ms")
        print(f"  TTFT P99: {best.metrics.get('ttft_p99', 0):.2f} ms")
        print(f"  Success Rate: {best.success_rate * 100:.1f}%")
        print("=" * 60)

    def get_results_df(self):
        """Get results as pandas DataFrame."""
        import pandas as pd

        rows = []
        for r in self.results:
            row = {
                "trial_id": r.trial_id,
                "score": r.score,
                "error": r.error,
                "gpu_memory_utilization": r.config.gpu_memory_utilization,
                "tensor_parallel": r.config.tensor_parallel,
                "max_model_len": r.config.max_model_len,
                "max_num_seqs": r.config.max_num_seqs,
                "tps": r.tps,
                "qps": r.metrics.get("qps", 0),
                "latency_p50": r.metrics.get("latency_p50", 0),
                "latency_p90": r.metrics.get("latency_p90", 0),
                "latency_p99": r.latency_p99,
                "ttft_p99": r.metrics.get("ttft_p99", 0),
                "success_rate": r.success_rate,
            }
            rows.append(row)

        return pd.DataFrame(rows)


def tune_vllm(
    model_path: str,
    gpu_ids: str,
    strategy: str = "bayesian",
    objective: str = "throughput",
    device: str = "nvidia",
    max_trials: int = 20,
    concurrency: int = 100,
    duration: int = 60,
    output_dir: str = "./results/autotune",
    verbose: bool = True,
) -> TuningResult:
    """Convenience function for quick tuning."""
    from .templates import save_tuning_report, generate_deploy_template

    tuner = AutoTuner(
        model_path=model_path,
        gpu_ids=gpu_ids,
        strategy=strategy,
        objective=objective,
        device=device,
        max_trials=max_trials,
        verbose=verbose,
    )

    result = tuner.run()

    if result:
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_tuning_report(tuner.results, output_path / "tuning_report.json")
        generate_deploy_template(result.config, output_path / "best_config.yaml")

        if verbose:
            print(f"\nResults saved to {output_path}")

    return result
