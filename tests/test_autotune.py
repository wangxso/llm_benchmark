"""Tests for Auto-Tuning Agent."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.autotune import (
    SearchSpace,
    ParameterRange,
    TuningConfig,
    TuningResult,
    Objective,
    RandomSearch,
    BayesianSearch,
    GridSearch,
    AutoTuner,
    generate_deploy_template,
    save_tuning_report,
)
from src.autotune.config import get_default_vllm_space, get_high_throughput_space, get_low_latency_space
from src.autotune.search import create_search_strategy


class TestParameterRange:
    """Test ParameterRange class."""

    def test_continuous_float_range(self):
        """Test continuous float parameter."""
        param = ParameterRange(name="test", min_val=0.1, max_val=0.9)
        assert param.param_type == "float"
        assert param.values is None

    def test_discrete_int_range(self):
        """Test discrete integer parameter."""
        param = ParameterRange(name="test", min_val=1, max_val=8, step=2)
        assert param.param_type == "int"

    def test_categorical_range(self):
        """Test categorical parameter."""
        param = ParameterRange(name="test", values=[1, 2, 4, 8])
        assert param.param_type == "categorical"
        assert param.values == [1, 2, 4, 8]


class TestSearchSpace:
    """Test SearchSpace class."""

    def test_default_constraints(self):
        """Test default constraint values."""
        space = SearchSpace(parameters=[])
        assert "latency_p99_ms" in space.constraints
        assert "success_rate" in space.constraints

    def test_validate_result_pass(self):
        """Test result validation - pass case."""
        space = SearchSpace(
            parameters=[],
            constraints={"latency_p99_ms": (0, 1000), "success_rate": (0.9, 1.0)},
        )
        metrics = {"latency_p99_ms": 500, "success_rate": 0.95}
        valid, violations = space.validate_result(metrics)
        assert valid is True
        assert len(violations) == 0

    def test_validate_result_fail(self):
        """Test result validation - fail case."""
        space = SearchSpace(
            parameters=[],
            constraints={"latency_p99_ms": (0, 500)},
        )
        metrics = {"latency_p99_ms": 1000}
        valid, violations = space.validate_result(metrics)
        assert valid is False
        assert len(violations) == 1


class TestTuningConfig:
    """Test TuningConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TuningConfig()
        assert config.gpu_memory_utilization == 0.85
        assert config.tensor_parallel == 1
        assert config.max_model_len == 4096
        assert config.max_num_seqs == 128

    def test_to_vllm_args(self):
        """Test conversion to vLLM arguments."""
        config = TuningConfig(
            gpu_memory_utilization=0.9,
            tensor_parallel=2,
            max_model_len=8192,
            max_num_seqs=256,
        )
        args = config.to_vllm_args()
        assert args["gpu_memory_utilization"] == 0.9
        assert args["tensor_parallel_size"] == 2
        assert args["max_model_len"] == 8192
        assert args["max_num_seqs"] == 256

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = TuningConfig()
        d = config.to_dict()
        assert "gpu_memory_utilization" in d
        assert "tensor_parallel" in d
        assert "max_model_len" in d


class TestTuningResult:
    """Test TuningResult class."""

    def test_calculate_score_throughput(self):
        """Test score calculation for throughput objective."""
        result = TuningResult(
            trial_id=1,
            config=TuningConfig(),
            metrics={"tps": 1000.0, "latency_p99_ms": 500.0},
        )
        score = result.calculate_score(Objective.THROUGHPUT)
        assert score == 1000.0

    def test_calculate_score_latency(self):
        """Test score calculation for latency objective."""
        result = TuningResult(
            trial_id=1,
            config=TuningConfig(),
            metrics={"tps": 1000.0, "latency_p99_ms": 500.0},
        )
        score = result.calculate_score(Objective.LATENCY)
        assert score == -500.0

    def test_calculate_score_balanced(self):
        """Test score calculation for balanced objective."""
        result = TuningResult(
            trial_id=1,
            config=TuningConfig(),
            metrics={"tps": 1000.0, "latency_p99_ms": 500.0},
        )
        score = result.calculate_score(Objective.BALANCED)
        assert score == 1000.0 * 100 / 500.0

    def test_error_result_has_inf_score(self):
        """Test that error results have -inf score."""
        result = TuningResult(trial_id=1, config=TuningConfig(), error="Failed")
        score = result.calculate_score(Objective.THROUGHPUT)
        assert score == float("-inf")


class TestSearchStrategies:
    """Test search strategies."""

    def test_random_search(self):
        """Test random search strategy."""
        strategy = RandomSearch(seed=42)
        study = strategy.create_study()

        space = SearchSpace(parameters=[
            ParameterRange(name="gpu_memory_utilization", min_val=0.7, max_val=0.9, step=0.1),
            ParameterRange(name="tensor_parallel", values=[1, 2]),
        ])

        trial = {}
        config = strategy.suggest_config(trial, space)

        assert 0.7 <= config.gpu_memory_utilization <= 0.9
        assert config.tensor_parallel in [1, 2]

    def test_grid_search(self):
        """Test grid search strategy."""
        strategy = GridSearch()

        space = SearchSpace(parameters=[
            ParameterRange(name="tensor_parallel", values=[1, 2]),
            ParameterRange(name="max_model_len", values=[4096, 8192]),
        ])

        strategy.set_search_space(space)
        assert strategy.total_combinations == 4

    def test_bayesian_search(self):
        """Test Bayesian search strategy."""
        strategy = BayesianSearch()
        study = strategy.create_study()

        assert study is not None
        assert strategy._study is not None


class TestSearchStrategyFactory:
    """Test search strategy factory."""

    def test_create_bayesian(self):
        """Test creating Bayesian strategy."""
        strategy = create_search_strategy("bayesian")
        assert isinstance(strategy, BayesianSearch)

    def test_create_random(self):
        """Test creating Random strategy."""
        strategy = create_search_strategy("random")
        assert isinstance(strategy, RandomSearch)

    def test_create_grid(self):
        """Test creating Grid strategy."""
        strategy = create_search_strategy("grid")
        assert isinstance(strategy, GridSearch)

    def test_invalid_strategy(self):
        """Test invalid strategy name."""
        with pytest.raises(ValueError):
            create_search_strategy("invalid")


class TestPredefinedSpaces:
    """Test predefined search spaces."""

    def test_default_space(self):
        """Test default VLLM search space."""
        space = get_default_vllm_space(gpu_count=1)
        assert len(space.parameters) == 4
        param_names = [p.name for p in space.parameters]
        assert "gpu_memory_utilization" in param_names
        assert "tensor_parallel" in param_names

    def test_default_space_multi_gpu(self):
        """Test default space with multiple GPUs."""
        space = get_default_vllm_space(gpu_count=4)
        tp_param = next(p for p in space.parameters if p.name == "tensor_parallel")
        assert tp_param.values == [1, 2, 4]

    def test_high_throughput_space(self):
        """Test high throughput search space."""
        space = get_high_throughput_space(gpu_count=4)
        assert space.constraints["latency_p99_ms"] == (0, 3000)

    def test_low_latency_space(self):
        """Test low latency search space."""
        space = get_low_latency_space(gpu_count=4)
        assert space.constraints["latency_p99_ms"][1] == 500


class TestTemplates:
    """Test template generation."""

    def test_generate_deploy_template(self, tmp_path):
        """Test deployment template generation."""
        config = TuningConfig(
            gpu_memory_utilization=0.9,
            tensor_parallel=2,
            max_model_len=8192,
        )

        template = generate_deploy_template(
            config,
            output_path=tmp_path / "config.yaml",
            model_path="/models/test",
            gpu_ids="0,1",
        )

        assert template["vllm"]["gpu_memory_utilization"] == 0.9
        assert template["vllm"]["tensor_parallel_size"] == 2
        assert "command" in template

    def test_save_tuning_report(self, tmp_path):
        """Test saving tuning report."""
        result = TuningResult(
            trial_id=1,
            config=TuningConfig(),
            metrics={"tps": 100.0, "latency_p99_ms": 500.0},
        )
        result.calculate_score(Objective.THROUGHPUT)
        results = [result]

        report = save_tuning_report(results, tmp_path / "report.json")

        assert report["summary"]["total_trials"] == 1
        assert (tmp_path / "report.csv").exists()


class TestAutoTuner:
    """Test AutoTuner class."""

    def test_initialization(self):
        """Test AutoTuner initialization."""
        tuner = AutoTuner(
            model_path="/models/test",
            gpu_ids="0",
            max_trials=5,
        )

        assert tuner.model_path == "/models/test"
        assert tuner.gpu_ids == "0"
        assert tuner.max_trials == 5
        assert tuner.objective == Objective.THROUGHPUT

    @pytest.mark.asyncio
    async def test_evaluate_config_mock(self):
        """Test configuration evaluation with mocked evaluator."""
        from src.autotune.evaluator import ConfigEvaluator

        evaluator = ConfigEvaluator(
            model_path="/models/test",
            gpu_ids="0",
            verbose=False,
        )

        # Mock the vLLM start and load test
        with patch.object(evaluator, '_start_instance', new_callable=AsyncMock) as mock_start:
            with patch.object(evaluator, '_run_load_test', new_callable=AsyncMock) as mock_load:
                with patch.object(evaluator, '_stop_instance', new_callable=AsyncMock):
                    mock_start.return_value = Mock(base_url="http://localhost:8100")
                    mock_load.return_value = {
                        "tps": 100.0,
                        "qps": 10.0,
                        "latency_p99_ms": 500.0,
                        "success_rate": 0.99,
                    }

                    config = TuningConfig()
                    result = await evaluator.evaluate(config, trial_id=1)

                    assert result.error is None
                    assert result.tps == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
