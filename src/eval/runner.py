"""Evaluation runner for executing benchmarks"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from tqdm import tqdm

from .prompts import format_prompt
from .scorer import extract_answer, score_results

if TYPE_CHECKING:
    from .base import BaseBenchmark


class EvalRunner:
    """Runner for executing benchmark evaluations

    Supports:
    - vLLM local server (OpenAI compatible)
    - Remote OpenAI-compatible API
    - Anthropic API
    """

    def __init__(
        self,
        benchmark: "BaseBenchmark",
        host: str = "localhost",
        port: int = 8000,
        model: Optional[str] = None,
        concurrency: int = 8,
        timeout: int = 60,
        hf_token: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_type: str = "openai",
    ):
        self.benchmark = benchmark
        self.host = host
        self.port = port
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout
        self.hf_token = hf_token
        self.api_key = api_key
        self.api_type = api_type.lower()

        # Determine base URL
        if api_base_url:
            self.base_url = api_base_url.rstrip("/")
        else:
            self.base_url = f"http://{host}:{port}/v1"

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API request"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            if self.api_type == "anthropic":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _send_request_openai(
        self,
        session,
        prompt: str,
    ) -> Dict[str, Any]:
        """Send request to OpenAI-compatible API"""
        import aiohttp

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.0,
        }
        if self.model:
            payload["model"] = self.model

        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    return {"success": False, "error": f"HTTP {response.status}: {text}"}

                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                return {"success": True, "response": content}

        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_request_anthropic(
        self,
        session,
        prompt: str,
    ) -> Dict[str, Any]:
        """Send request to Anthropic API"""
        import aiohttp

        payload = {
            "model": self.model or "claude-3-haiku-20240307",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            # Anthropic API endpoint
            url = self.base_url if "/messages" in self.base_url else f"{self.base_url}/messages"
            if "api.anthropic.com" in self.base_url and "/v1/messages" not in self.base_url:
                url = "https://api.anthropic.com/v1/messages"

            async with session.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    return {"success": False, "error": f"HTTP {response.status}: {text}"}

                data = await response.json()
                # Anthropic response format
                if "content" in data and isinstance(data["content"], list):
                    content = data["content"][0].get("text", "")
                else:
                    content = data.get("content", "")
                return {"success": True, "response": content}

        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_request(
        self,
        session,
        prompt: str,
    ) -> Dict[str, Any]:
        """Send a single request based on API type"""
        if self.api_type == "anthropic":
            return await self._send_request_anthropic(session, prompt)
        else:
            return await self._send_request_openai(session, prompt)

    async def _evaluate_item(
        self,
        session,
        item: Dict,
        prompt_style: str,
        semaphore: asyncio.Semaphore,
    ) -> Dict:
        """Evaluate a single item"""
        async with semaphore:
            prompt = format_prompt(
                item["question"],
                item["choices"],
                style=prompt_style,
            )

            result = await self._send_request(session, prompt)

            if result["success"]:
                num_options = len(item.get("choices", []))
                predicted = extract_answer(result["response"], num_options=num_options)
                return {
                    "question": item["question"][:100] + "...",  # Truncate for report
                    "choices": item["choices"],
                    "actual": item["answer"],
                    "predicted": predicted,
                    "subject": item.get("subject", "unknown"),
                    "response": result["response"][:200],
                    "success": True,
                }
            else:
                return {
                    "question": item["question"][:100] + "...",
                    "actual": item["answer"],
                    "predicted": None,
                    "subject": item.get("subject", "unknown"),
                    "error": result.get("error"),
                    "success": False,
                }

    async def run_async(
        self,
        prompt_style: str = "zero_shot",
        max_samples: Optional[int] = None,
        subject: Optional[str] = None,
    ) -> tuple:
        """Run the evaluation asynchronously"""
        import aiohttp

        # Load dataset
        items = self.benchmark.load(
            subject=subject,
            max_samples=max_samples,
            token=self.hf_token,
        )

        if not items:
            error_report = {
                "error": "No items to evaluate",
                "benchmark": self.benchmark.name,
                "overall_accuracy": 0.0,
                "total_questions": 0,
                "correct": 0,
                "subjects": {},
            }
            return error_report, []

        # Get model info if not specified
        if not self.model and self.api_type == "openai":
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/models",
                        headers=self._get_headers(),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("data"):
                                self.model = data["data"][0].get("id", "unknown")
            except Exception:
                self.model = "unknown"

        if not self.model:
            self.model = "unknown"

        # Run evaluation with concurrency
        semaphore = asyncio.Semaphore(self.concurrency)
        results = []

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._evaluate_item(session, item, prompt_style, semaphore)
                for item in items
            ]

            for f in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"Evaluating {self.benchmark.name}",
            ):
                result = await f
                results.append(result)

        # Calculate metrics
        metrics = score_results([r for r in results if r.get("success")])

        # Calculate category metrics if benchmark has categories
        category_map = getattr(self.benchmark, "get_category_map", lambda: None)()
        if category_map:
            from .scorer import score_results_with_categories
            metrics_with_cat = score_results_with_categories(
                [r for r in results if r.get("success")],
                category_map=category_map,
            )
            metrics["categories"] = metrics_with_cat.get("categories", {})

        report = {
            "benchmark": self.benchmark.name,
            "model": self.model,
            "prompt_style": prompt_style,
            "overall_accuracy": metrics["overall_accuracy"],
            "total_questions": metrics["total_questions"],
            "correct": metrics["correct"],
            "subjects": metrics["subjects"],
            "categories": metrics.get("categories", {}),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "api_type": self.api_type,
                "base_url": self.base_url,
                "concurrency": self.concurrency,
            },
        }

        return report, results

    def run(
        self,
        prompt_style: str = "zero_shot",
        max_samples: Optional[int] = None,
        subject: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Run the evaluation (sync wrapper)"""
        report, results = asyncio.run(
            self.run_async(
                prompt_style=prompt_style,
                max_samples=max_samples,
                subject=subject,
            )
        )

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = output_path / f"eval_{self.benchmark.name}_{timestamp}.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            details_file = output_path / f"eval_{self.benchmark.name}_{timestamp}_details.json"
            with open(details_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            report["report_file"] = str(report_file)
            report["details_file"] = str(details_file)

        return report
