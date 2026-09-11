"""Local-only tokenizer rendering and vLLM generation backend."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Protocol

from .environment import guard_local_inference


def derive_seed(run_id: str, pattern_id: str, sample_index: int) -> int:
    digest = hashlib.sha256(f"{run_id}||{pattern_id}||{sample_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RenderedPrompt:
    raw_user_text: str
    rendered: str
    raw_token_count: int
    rendered_token_count: int
    raw_hash: str
    rendered_hash: str
    tokenizer_hash: str
    chat_template_hash: str


def render_user_prompt(tokenizer: Any, text: str, tokenizer_revision: str) -> RenderedPrompt:
    messages = [{"role": "user", "content": text}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    raw_tokens = tokenizer.encode(text, add_special_tokens=False)
    rendered_tokens = tokenizer.encode(rendered, add_special_tokens=False)
    template = str(getattr(tokenizer, "chat_template", ""))
    tokenizer_fingerprint = sha256_text(
        f"{getattr(tokenizer, 'name_or_path', '')}\n{tokenizer_revision}\n{len(tokenizer)}"
    )
    return RenderedPrompt(
        raw_user_text=text,
        rendered=rendered,
        raw_token_count=len(raw_tokens),
        rendered_token_count=len(rendered_tokens),
        raw_hash=sha256_text(text),
        rendered_hash=sha256_text(rendered),
        tokenizer_hash=tokenizer_fingerprint,
        chat_template_hash=sha256_text(template),
    )


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    token_ids: list[int] | None = None


class Backend(Protocol):
    def generate(self, prompts: list[str], seeds: list[int], parameters: dict[str, Any]) -> list[Completion]: ...


class MockBackend:
    """Deterministic backend used only for CPU tests and dry pipeline exercises."""

    def generate(self, prompts: list[str], seeds: list[int], parameters: dict[str, Any]) -> list[Completion]:
        output = []
        for prompt, seed in zip(prompts, seeds, strict=True):
            token = random.Random(seed).choice(("alpha", "bravo", "charlie", "delta"))
            text = f"Mock local response {token} ({sha256_text(prompt)[:8]}; seed={seed})."
            token_ids = [int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") for word in text.split()]
            output.append(Completion(text, "stop", len(prompt.split()), len(token_ids), token_ids))
        return output


class VLLMBackend:
    def __init__(self, profile: dict[str, Any]):
        guard_local_inference(profile)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM is not installed; run scripts/bootstrap_l4.sh") from exc
        self._sampling_cls = SamplingParams
        kwargs = {
            "model": profile["model"],
            "tokenizer": profile.get("tokenizer", profile["model"]),
            "max_model_len": profile["max_model_len"],
            "gpu_memory_utilization": profile["gpu_memory_utilization"],
            "tensor_parallel_size": 1,
            "trust_remote_code": False,
            "enable_prefix_caching": False,
            "max_num_seqs": profile.get("max_num_seqs", 1),
            "language_model_only": profile.get("language_model_only", True),
        }
        if "generation_config" in profile:
            kwargs["generation_config"] = profile["generation_config"]
        if "seed" in profile:
            kwargs["seed"] = profile["seed"]
        from pathlib import Path

        if not Path(profile["model"]).exists():
            kwargs["revision"] = profile["revision"]
        if not Path(profile.get("tokenizer", profile["model"])).exists():
            kwargs["tokenizer_revision"] = profile.get("tokenizer_revision", profile["revision"])
        if profile.get("cpu_offload_gb"):
            kwargs["cpu_offload_gb"] = profile["cpu_offload_gb"]
        if profile.get("quantization") not in (None, "fp8"):
            kwargs["quantization"] = profile["quantization"]
        self.engine = LLM(**kwargs)

    def generate(self, prompts: list[str], seeds: list[int], parameters: dict[str, Any]) -> list[Completion]:
        results: list[Completion] = []
        for prompt, seed in zip(prompts, seeds, strict=True):
            sampling = self._sampling_cls(
                temperature=parameters["temperature"],
                top_p=parameters["top_p"],
                max_tokens=parameters["max_tokens"],
                seed=seed,
            )
            request = self.engine.generate([prompt], sampling, use_tqdm=False)[0]
            candidate = request.outputs[0]
            results.append(
                Completion(
                    text=candidate.text,
                    finish_reason=str(candidate.finish_reason) if candidate.finish_reason else None,
                    prompt_tokens=len(request.prompt_token_ids),
                    completion_tokens=len(candidate.token_ids),
                    token_ids=list(candidate.token_ids),
                )
            )
        return results


def make_backend(profile: dict[str, Any], mock: bool = False) -> Backend:
    return MockBackend() if mock or profile.get("backend") == "mock" else VLLMBackend(profile)
