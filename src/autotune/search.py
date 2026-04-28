"""Search strategies for Auto-Tuning Agent."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import random

if TYPE_CHECKING:
    import optuna
    from .config import SearchSpace, TuningConfig, TuningResult


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def create_study(self, direction: str = "maximize") -> Any:
        """Create optimization study."""
        pass

    @abstractmethod
    def suggest_config(self, trial: Any, space: "SearchSpace") -> "TuningConfig":
        """Suggest configuration for given trial."""
        pass

    @abstractmethod
    def update(self, trial: Any, result: "TuningResult"):
        """Update study with result."""
        pass

    @abstractmethod
    def get_best_trial(self) -> Optional[Any]:
        """Get the best trial so far."""
        pass


class RandomSearch(SearchStrategy):
    """Random search strategy - baseline method."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._trials: List[Dict[str, Any]] = []
        self._results: List["TuningResult"] = []
        self._trial_count = 0
        random.seed(seed)

    def create_study(self, direction: str = "maximize") -> Dict[str, Any]:
        """Create a simple study object."""
        self._direction = direction
        return {
            "direction": direction,
            "trials": self._trials,
        }

    def suggest_config(self, trial: Any, space: "SearchSpace") -> "TuningConfig":
        """Randomly sample configuration from search space."""
        from .config import TuningConfig

        config_params = {}
        for param in space.parameters:
            if param.values is not None:
                config_params[param.name] = random.choice(param.values)
            elif param.param_type == "int":
                min_val = int(param.min_val)
                max_val = int(param.max_val)
                step = int(param.step) if param.step else 1
                values = list(range(min_val, max_val + 1, step))
                config_params[param.name] = random.choice(values)
            else:
                config_params[param.name] = random.uniform(
                    param.min_val, param.max_val
                )
                if param.step:
                    # Quantize to step
                    config_params[param.name] = round(
                        config_params[param.name] / param.step
                    ) * param.step

        self._trial_count += 1
        trial["params"] = config_params
        trial["trial_id"] = self._trial_count
        self._trials.append(trial)

        return TuningConfig(**config_params)

    def update(self, trial: Any, result: "TuningResult"):
        """Store result for tracking."""
        trial["value"] = result.score
        trial["result"] = result
        self._results.append(result)

    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """Get trial with best score."""
        if not self._trials:
            return None
        return max(
            [t for t in self._trials if "value" in t],
            key=lambda t: t.get("value", float("-inf")),
            default=None
        )


class BayesianSearch(SearchStrategy):
    """Bayesian optimization using Optuna's TPE sampler."""

    def __init__(
        self,
        n_startup_trials: int = 5,
        seed: int = 42,
    ):
        self.n_startup_trials = n_startup_trials
        self.seed = seed
        self._study: Optional["optuna.Study"] = None

    def create_study(self, direction: str = "maximize") -> "optuna.Study":
        """Create Optuna study with TPE sampler."""
        import optuna
        from optuna.samplers import TPESampler

        sampler = TPESampler(
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
        )

        self._study = optuna.create_study(
            direction=direction,
            sampler=sampler,
        )
        return self._study

    def suggest_config(self, trial: "optuna.Trial", space: "SearchSpace") -> "TuningConfig":
        """Sample configuration using Optuna trial."""
        from .config import TuningConfig

        config_params = {}
        for param in space.parameters:
            config_params[param.name] = param.sample_value(trial)

        return TuningConfig(**config_params)

    def update(self, trial: "optuna.Trial", result: "TuningResult"):
        """Optuna handles this via trial.report() during objective function."""
        pass  # Optuna handles this internally

    def get_best_trial(self) -> Optional["optuna.trial.FrozenTrial"]:
        """Get best trial from study."""
        if self._study is None:
            return None
        return self._study.best_trial


class GridSearch(SearchStrategy):
    """Grid search - exhaustive enumeration."""

    def __init__(self):
        self._combinations: List[Dict[str, Any]] = []
        self._current_index = 0
        self._trials: List[Dict[str, Any]] = []
        self._results: List["TuningResult"] = []

    def _generate_combinations(self, space: "SearchSpace") -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools

        param_values = []
        for param in space.parameters:
            if param.values is not None:
                param_values.append([(param.name, v) for v in param.values])
            elif param.param_type == "int":
                min_val = int(param.min_val)
                max_val = int(param.max_val)
                step = int(param.step) if param.step else 1
                values = list(range(min_val, max_val + 1, step))
                param_values.append([(param.name, v) for v in values])
            else:
                # For float, use step to discretize
                values = []
                v = param.min_val
                while v <= param.max_val:
                    values.append(v)
                    v += param.step if param.step else 0.1
                param_values.append([(param.name, round(v, 4)) for v in values])

        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(combo))

        return combinations

    def create_study(self, direction: str = "maximize") -> Dict[str, Any]:
        """Initialize grid search."""
        self._direction = direction
        return {
            "direction": direction,
            "total_combinations": len(self._combinations),
            "current_index": self._current_index,
        }

    def set_search_space(self, space: "SearchSpace"):
        """Set search space and generate grid."""
        self._combinations = self._generate_combinations(space)
        self._current_index = 0

    def suggest_config(self, trial: Any, space: "SearchSpace") -> "TuningConfig":
        """Get next configuration from grid."""
        from .config import TuningConfig

        if not self._combinations:
            self.set_search_space(space)

        if self._current_index >= len(self._combinations):
            raise StopIteration("All grid combinations exhausted")

        params = self._combinations[self._current_index]
        trial["params"] = params
        trial["trial_id"] = self._current_index + 1
        self._trials.append(trial)

        self._current_index += 1
        return TuningConfig(**params)

    def update(self, trial: Any, result: "TuningResult"):
        """Store result."""
        trial["value"] = result.score
        trial["result"] = result
        self._results.append(result)

    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """Get trial with best score."""
        if not self._trials:
            return None
        return max(
            [t for t in self._trials if "value" in t],
            key=lambda t: t.get("value", float("-inf")),
            default=None
        )

    @property
    def total_combinations(self) -> int:
        """Total number of grid combinations."""
        return len(self._combinations)


def create_search_strategy(
    strategy: str,
    n_startup_trials: int = 5,
    seed: int = 42,
) -> SearchStrategy:
    """Factory function to create search strategy."""
    if strategy == "bayesian":
        return BayesianSearch(n_startup_trials=n_startup_trials, seed=seed)
    elif strategy == "random":
        return RandomSearch(seed=seed)
    elif strategy == "grid":
        return GridSearch()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
