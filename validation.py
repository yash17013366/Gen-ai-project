
import re
from collections import Counter
import time

import requests


OLLAMA_URL = "http://127.0.0.1:11434"


def calculate_answer_scores(answer: str, expected_answer: str) -> dict[str, float | bool]:
    """Compare an answer with a reference using normalized exact match and token F1."""
    answer_tokens = re.findall(r"\w+", answer.casefold())
    expected_tokens = re.findall(r"\w+", expected_answer.casefold())
    exact_match = answer_tokens == expected_tokens

    if not answer_tokens or not expected_tokens:
        token_f1 = float(exact_match)
    else:
        overlap = sum((Counter(answer_tokens) & Counter(expected_tokens)).values())
        precision = overlap / len(answer_tokens)
        recall = overlap / len(expected_tokens)
        token_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    return {
        "exact_match": exact_match,
        "token_f1": token_f1,
    }


def summarize_benchmark(scores: list[dict[str, float | bool]]) -> dict[str, float | int]:
    """Calculate strict exact-match accuracy and macro token F1 for a benchmark."""
    if not scores:
        raise ValueError("At least one scored answer is required.")

    return {
        "question_count": len(scores),
        "exact_match_accuracy": (
            sum(score["exact_match"] is True for score in scores)
            / len(scores)
        ),
        "mean_token_f1": (
            sum(float(score["token_f1"]) for score in scores)
            / len(scores)
        ),
    }


def calculate_performance_metrics(
    result: dict,
    request_latency_seconds: float
) -> dict[str, float | int | None]:
    """Extract Ollama timings and derive prompt/generation throughput."""
    metrics: dict[str, float | int | None] = {
        "request_latency_seconds": request_latency_seconds,
    }

    duration_fields = {
        "ollama_total_seconds": "total_duration",
        "model_load_seconds": "load_duration",
        "prompt_processing_seconds": "prompt_eval_duration",
        "generation_seconds": "eval_duration",
    }
    for metric_name, result_name in duration_fields.items():
        duration_ns = result.get(result_name)
        metrics[metric_name] = (
            duration_ns / 1_000_000_000
            if isinstance(duration_ns, (int, float)) and duration_ns >= 0
            else None
        )

    count_fields = {
        "prompt_tokens": "prompt_eval_count",
        "output_tokens": "eval_count",
    }
    for metric_name, result_name in count_fields.items():
        token_count = result.get(result_name)
        metrics[metric_name] = (
            token_count
            if isinstance(token_count, int) and token_count >= 0
            else None
        )

    prompt_tokens = metrics["prompt_tokens"]
    prompt_seconds = metrics["prompt_processing_seconds"]
    metrics["prompt_tokens_per_second"] = (
        prompt_tokens / prompt_seconds
        if isinstance(prompt_tokens, int)
        and isinstance(prompt_seconds, float)
        and prompt_seconds > 0
        else None
    )

    output_tokens = metrics["output_tokens"]
    generation_seconds = metrics["generation_seconds"]
    metrics["tokens_per_second"] = (
        output_tokens / generation_seconds
        if isinstance(output_tokens, int)
        and isinstance(generation_seconds, float)
        and generation_seconds > 0
        else None
    )

    return metrics


def validate_ollama():
    """Check whether the Ollama API is responding."""
    print("\n========== SYSTEM VALIDATION ==========")

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )
        response.raise_for_status()

        models = response.json().get("models", [])
        print("[PASS] Ollama API is responding")

        if models:
            print("[PASS] Installed models:")
            for model in models:
                print(f"  - {model['name']}")
        else:
            print("[WARN] No Ollama models installed")

        return True

    except requests.RequestException as error:
        print(f"[FAIL] Ollama connection: {error}")
        return False


def validate_model(model_name):
    """Check whether the requested model is installed."""
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )
        response.raise_for_status()

        models = response.json().get("models", [])
        installed = [
            model["name"] for model in models
        ]

        if model_name in installed:
            print(f"[PASS] Model available: {model_name}")
            return True

        print(f"[FAIL] Model not found: {model_name}")
        print(f"[INFO] Available models: {installed}")
        return False

    except requests.RequestException as error:
        print(f"[FAIL] Model validation failed: {error}")
        return False


def measure_model_latency(model_name, prompt):
    """Run a prompt and print Ollama's timing and throughput measurements."""
    print("\n========== LLM PERFORMANCE ==========")

    try:
        start = time.perf_counter()

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        elapsed = time.perf_counter() - start
        response.raise_for_status()
        result = response.json()
        metrics = calculate_performance_metrics(result, elapsed)

        print("[PASS] LLM response generated")
        print(f"Request latency: {metrics['request_latency_seconds']:.3f} seconds")
        for label, metric_name in (
            ("Ollama total time", "ollama_total_seconds"),
            ("Model load time", "model_load_seconds"),
            ("Prompt processing time", "prompt_processing_seconds"),
            ("Generation time", "generation_seconds"),
        ):
            value = metrics[metric_name]
            if value is not None:
                print(f"{label}: {value:.3f} seconds")

        for label, metric_name in (
            ("Prompt tokens", "prompt_tokens"),
            ("Output tokens", "output_tokens"),
        ):
            value = metrics[metric_name]
            if value is not None:
                print(f"{label}: {value}")

        for label, metric_name in (
            ("Prompt processing throughput", "prompt_tokens_per_second"),
            ("Generation throughput", "tokens_per_second"),
        ):
            value = metrics[metric_name]
            if value is not None:
                print(f"{label}: {value:.2f} tokens/sec")

        return result.get("response", "")

    except requests.RequestException as error:
        print(f"[FAIL] LLM request failed: {error}")
        return None
