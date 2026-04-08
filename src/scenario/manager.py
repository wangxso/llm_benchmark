from typing import Dict, Any, List, Optional
from datetime import datetime


class ScenarioManager:
    """Scenario manager for test orchestration"""

    def __init__(self, config: Dict):
        self.config = config
        self.current_scenario = None
        self.experiment_history: List[Dict] = []

    def create_scenario(self) -> Dict[str, Any]:
        """Create a test scenario"""
        scenario = {
            "name": self.config.get("scenario", {}).get("name", "default"),
            "model": self.config.get("vllm", {}).get("model"),
            "backend": "vllm",
            "tensor_parallel": self.config.get("vllm", {}).get("tensor_parallel", 1),
            "load_type": self.config.get("load", {}).get("type", "fixed"),
            "concurrency": self.config.get("load", {}).get("base_concurrency", 100),
            "duration": self.config.get("load", {}).get("duration", 300),
            "warmup": self.config.get("load", {}).get("warmup_duration", 10),
            "created_at": datetime.now().isoformat(),
        }

        self.current_scenario = scenario
        return scenario

    def record_experiment(self, results: Dict):
        """Record experiment results"""
        experiment = {
            "scenario": self.current_scenario,
            "results": results,
            "recorded_at": datetime.now().isoformat(),
        }

        self.experiment_history.append(experiment)

    def get_history(self) -> List[Dict]:
        """Get experiment history"""
        return self.experiment_history

    def compare_experiments(self, exp1_idx: int, exp2_idx: int) -> Dict[str, Any]:
        """Compare two experiments"""
        if exp1_idx >= len(self.experiment_history) or exp2_idx >= len(
            self.experiment_history
        ):
            return {"error": "Invalid experiment index"}

        exp1 = self.experiment_history[exp1_idx]
        exp2 = self.experiment_history[exp2_idx]

        return {
            "exp1": exp1,
            "exp2": exp2,
            "comparison": self._calculate_comparison(exp1["results"], exp2["results"]),
        }

    def _calculate_comparison(self, r1: Dict, r2: Dict) -> Dict[str, Any]:
        """Calculate metrics comparison"""
        comparison = {}

        for key in ["qps", "tps", "latency_p50", "latency_p99", "success_rate"]:
            if key in r1 and key in r2:
                if r2[key] != 0:
                    change = ((r1[key] - r2[key]) / r2[key]) * 100
                    comparison[key] = {
                        "exp1": r1[key],
                        "exp2": r2[key],
                        "change_percent": change,
                    }

        return comparison
