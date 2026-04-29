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
        rate_limit: float = 0,
        timeout: int = 60,
        offline: bool = False,
        hf_token: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_type: str = "openai",
        dataset_source: str = "huggingface",  # "huggingface" or "modelscope"
    ):
        self.benchmark = benchmark
        self.host = host
        self.port = port
        self.model = model
        self.concurrency = concurrency
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.offline = offline
        self.hf_token = hf_token
        self.api_key = api_key
        self.api_type = api_type.lower()
        self.dataset_source = dataset_source

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
            "max_tokens": 4000,  # Enough for CoT reasoning
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
                    # Try to extract error message from JSON before truncating
                    detail = ""
                    try:
                        err_data = json.loads(text)
                        err_obj = err_data.get("error", {})
                        if isinstance(err_obj, dict):
                            detail = err_obj.get("message", "")
                        elif err_obj:
                            detail = str(err_obj)
                    except:
                        pass
                    err_summary = f"HTTP {response.status}"
                    if detail:
                        err_summary += f": {detail}"
                    elif text:
                        err_summary += f": {text[:200]}"
                    return {"success": False, "error": err_summary}

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
            "max_tokens": 4000,  # Enough for CoT reasoning
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            # Determine the correct URL
            # User might provide:
            # 1. Full endpoint URL (contains /messages or /chat/completions)
            # 2. Base URL (we append /v1/messages for Anthropic)
            if "/messages" in self.base_url or "/chat/completions" in self.base_url:
                # User provided full URL, use as-is
                url = self.base_url
            elif "api.anthropic.com" in self.base_url:
                # Official Anthropic API
                url = "https://api.anthropic.com/v1/messages"
            elif self.base_url.rstrip("/").endswith("/anthropic"):
                # MiniMax and similar providers: base_url ends with /anthropic
                # Need to append /v1/messages
                url = f"{self.base_url.rstrip('/')}/v1/messages"
            else:
                # For other Anthropic-compatible providers
                # Try without appending /messages first, then with /messages
                url = self.base_url

            async with session.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                # If 404, try appending /messages
                if response.status == 404 and "/messages" not in url:
                    new_url = f"{url.rstrip('/')}/messages"
                    async with session.post(
                        new_url,
                        json=payload,
                        headers=self._get_headers(),
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as retry_response:
                        if retry_response.status == 200:
                            data = await retry_response.json()
                            if "content" in data and isinstance(data["content"], list):
                                content = data["content"][0].get("text", "")
                            else:
                                content = data.get("content", "")
                            return {"success": True, "response": content}
                        response = retry_response

                if response.status != 200:
                    text = await response.text()
                    # Try to extract error message from JSON before truncating
                    detail = ""
                    try:
                        err_data = json.loads(text)
                        err_obj = err_data.get("error", {})
                        if isinstance(err_obj, dict):
                            detail = err_obj.get("message", "")
                        elif err_obj:
                            detail = str(err_obj)
                    except:
                        pass
                    err_summary = f"HTTP {response.status}"
                    if detail:
                        err_summary += f": {detail}"
                    elif text:
                        err_summary += f": {text[:200]}"
                    return {"success": False, "error": err_summary}

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
                category=item.get("subject", ""),
            )

            result = await self._send_request(session, prompt)

            if result["success"]:
                num_options = len(item.get("choices", []))
                predicted = extract_answer(result["response"], num_options=num_options)
                return {
                    "question": item["question"],  # Keep full question
                    "choices": item["choices"],
                    "actual": item["answer"],
                    "predicted": predicted,
                    "subject": item.get("subject", "unknown"),
                    "response": result["response"],  # Keep full response
                    "success": True,
                }
            else:
                return {
                    "question": item["question"],
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
        stop_event=None,
    ) -> tuple:
        """Run the evaluation asynchronously"""
        import aiohttp

        # Load dataset
        items = self.benchmark.load(
            subject=subject,
            max_samples=max_samples,
            token=self.hf_token,
            offline=self.offline,
            source=self.dataset_source,
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

        # Rate limiter with lock for thread safety
        # Rate limit is in RPM (requests per minute), convert to seconds
        rate_lock = asyncio.Lock()
        min_interval = 60.0 / self.rate_limit if self.rate_limit > 0 else 0

        async def rate_limited_evaluate(session, item):
            """Wrap evaluation with rate limiting"""
            if min_interval > 0:
                async with rate_lock:
                    await asyncio.sleep(min_interval)
            return await self._evaluate_item(session, item, prompt_style, semaphore)

        async with aiohttp.ClientSession() as session:
            tasks = [
                rate_limited_evaluate(session, item)
                for item in items
            ]

            for f in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"Evaluating {self.benchmark.name}",
            ):
                if stop_event and stop_event.is_set():
                    break
                result = await f
                results.append(result)

        # Separate successful and failed results
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]

        # Calculate metrics only from successful results
        metrics = score_results(successful_results)

        # Calculate error statistics with detailed reasons
        attempted_count = len(results)
        successful_count = len(successful_results)
        failed_count = len(failed_results)
        error_types = {}
        error_details = {}  # Store detailed error messages

        for r in results:
            if not r.get("success") and r.get("error"):
                err_msg = r["error"]

                # Categorize error by HTTP status or keywords
                if "HTTP 401" in err_msg or "authentication" in err_msg.lower():
                    key = "Authentication Error"
                elif "HTTP 403" in err_msg:
                    key = "Forbidden"
                elif "HTTP 404" in err_msg:
                    key = "Not Found (check API endpoint)"
                elif "HTTP 429" in err_msg or "rate_limit" in err_msg.lower():
                    key = "Rate Limited"
                elif "Timeout" in err_msg:
                    key = "Timeout"
                elif "Connection" in err_msg:
                    key = "Connection Error"
                elif "overloaded" in err_msg or "529" in err_msg:
                    key = "Server Overloaded"
                else:
                    key = "Other"

                error_types[key] = error_types.get(key, 0) + 1

                # Extract detail message (after the HTTP status)
                if ": " in err_msg:
                    detail = err_msg.split(": ", 1)[1]
                else:
                    detail = err_msg

                # Store detailed message for this error type
                if key not in error_details:
                    error_details[key] = {}
                # Use first 100 chars of detail as key to group similar errors
                detail_key = detail[:100]
                error_details[key][detail_key] = error_details[key].get(detail_key, 0) + 1

        # Calculate category metrics if benchmark has categories
        category_map = getattr(self.benchmark, "get_category_map", lambda: None)()
        if category_map:
            from .scorer import score_results_with_categories
            metrics_with_cat = score_results_with_categories(
                successful_results,
                category_map=category_map,
            )
            metrics["categories"] = metrics_with_cat.get("categories", {})

        report = {
            "benchmark": self.benchmark.name,
            "model": self.model,
            "prompt_style": prompt_style,
            "attempted_count": attempted_count,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "overall_accuracy": metrics["overall_accuracy"],
            "total_questions": metrics["total_questions"],
            "correct": metrics["correct"],
            "subjects": metrics["subjects"],
            "categories": metrics.get("categories", {}),
            "error_types": error_types,
            "error_details": error_details,
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
        stop_event=None,
    ) -> Dict:
        """Run the evaluation (sync wrapper)"""
        report, results = asyncio.run(
            self.run_async(
                prompt_style=prompt_style,
                max_samples=max_samples,
                subject=subject,
                stop_event=stop_event,
            )
        )

        # Include results in report for immediate display
        report["details"] = [
            {
                "question": r.get("question", ""),
                "actual": r.get("actual"),
                "predicted": r.get("predicted"),
                "subject": r.get("subject", "unknown"),
                "response": r.get("response", ""),
                "success": r.get("success", False),
                "error": r.get("error"),
            }
            for r in results
        ]

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
