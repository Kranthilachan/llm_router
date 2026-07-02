"""
Optional model execution layer for the /route endpoint.

Routing remains the source of truth for model selection. This module only
decides whether a routed plan is executable in the current deployment and,
when it is, calls the configured provider adapter.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.models.schemas import RoutingDecision

# Router model names are synthetic in this repo. Only explicitly mapped names
# are executable through the real provider adapter.
OPENAI_MODEL_MAP = {
    "o3": "o3",
    "GPT-4o": "gpt-4o",
    "GPT-4o-Mini": "gpt-4o-mini",
}

_EXECUTABLE_TEXT_EXTENSIONS = {".txt", ".md"}
_SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_SUPPORTED_EXTENSIONS = {
    ".pdf",
    *_EXECUTABLE_TEXT_EXTENSIONS,
    *_SUPPORTED_AUDIO_EXTENSIONS,
    *_SUPPORTED_IMAGE_EXTENSIONS,
}
# Per-file safety caps (generous). Real budget check happens post-extraction.
# Map-reduce handles anything that exceeds the model window.
_PDF_TEXT_CHAR_LIMIT = 5_000_000
_TEXT_FILE_CHAR_LIMIT = 5_000_000
_TEXT_CONTEXT_CHARS = 200_000
_AUDIO_TRANSCRIPTION_MAX_BYTES = 24 * 1024 * 1024  # best-effort guard
_IMAGE_MAX_BYTES = 20 * 1024 * 1024  # OpenAI vision per-image limit

# Token/budget accounting. Rough English char→token ratio; conservative on purpose.
_CHARS_PER_TOKEN = 4
_DEFAULT_CONTEXT_WINDOW = 8192
_OUTPUT_RESERVE_TOKENS = 2000  # leave room for the model's response
_BUDGET_UTILIZATION = 0.8      # use 80% of usable input window, keep headroom
_MIN_CONTEXT_TOKENS = 1000     # floor so tiny models still get some context

# Map-reduce chunking for documents larger than the model's window.
_MAP_REDUCE_MIN_CHARS = 60_000        # do not bother chunking below this
_MAP_REDUCE_MAX_CHUNKS = 20           # hard cap to bound cost + latency
_MAP_CHUNK_BUDGET_SHARE = 0.55        # each map call uses 55% of usable budget
_MAP_INTERMEDIATE_RESERVE_TOKENS = 800  # per-part extract soft budget guidance


@dataclass
class ExecutionResult:
    status: str
    answer_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[Dict[str, str]] = None


def executable_model_names() -> List[str]:
    """Return the routed model names that can be executed in this deployment."""
    return sorted(OPENAI_MODEL_MAP)


def maybe_execute_routed_answer(
    *,
    decision: RoutingDecision,
    prompt: str,
    input_formats: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
) -> ExecutionResult:
    """
    Execute the routed primary model when the decision is safe and supported.

    v1 constraints:
      - single-model only
      - no verifier execution
      - no fallback failover
    """
    if decision.abstain:
        return ExecutionResult(status="abstained")
    if decision.escalate_to_human:
        return ExecutionResult(
            status="escalated",
            warnings=["Routing decision requires human review; execution skipped."],
        )

    plan = decision.selected_plan
    if plan is None:
        return ExecutionResult(
            status="unsupported",
            warnings=["No execution plan was selected."],
        )
    if plan.plan_type != "single_model":
        return ExecutionResult(
            status="unsupported",
            warnings=[f"Plan type '{plan.plan_type}' is not executable in v1."],
        )

    selected_model = plan.selected_model
    api_model = OPENAI_MODEL_MAP.get(selected_model or "")
    if not api_model:
        return ExecutionResult(
            status="unsupported",
            warnings=[f"Routed model '{selected_model}' is not mapped to an executable provider model."],
        )
    if not os.getenv("OPENAI_API_KEY"):
        return ExecutionResult(
            status="failed",
            error={"code": "missing_openai_api_key", "message": "OPENAI_API_KEY is required for execution."},
        )

    normalized_formats = [fmt.lower() for fmt in (input_formats or ["text"])]
    requested_non_text_formats = [fmt for fmt in normalized_formats if fmt != "text"]
    file_paths = [p for p in (files or []) if p]

    file_context: List[Dict[str, str]] = []
    file_images: List[Dict[str, str]] = []
    file_warnings: List[str] = []
    if file_paths:
        file_context, file_images, file_warnings = _extract_file_contexts(file_paths)

    if file_paths and not file_context and not file_images:
        return ExecutionResult(
            status="unsupported",
            warnings=["Execution cannot proceed: no extractable content found from uploaded file(s)."] + file_warnings,
        )

    # Non-text formats without any usable uploaded content stay unsupported.
    if requested_non_text_formats and not (file_paths and (file_context or file_images)):
        details: List[str] = []
        if file_paths:
            details.append("files are present")
        if requested_non_text_formats:
            details.append(f"non-text formats requested: {sorted(set(requested_non_text_formats))}")
        if file_warnings:
            details.extend(file_warnings)
        return ExecutionResult(
            status="unsupported",
            warnings=[f"Execution needs an extractable file for non-text formats; {', '.join(details)}."],
        )

    budget = _compute_context_char_budget(selected_model, prompt)
    extracted_chars_before = sum(len(item.get("text", "")) for item in file_context)
    usable_chars = budget["usable_context_chars"]

    if file_paths and file_warnings:
        file_warnings = [f"Upload extraction warning: {w}" for w in file_warnings]

    use_map_reduce = bool(
        file_context
        and extracted_chars_before > usable_chars
        and extracted_chars_before >= _MAP_REDUCE_MIN_CHARS
    )

    start = time.time()
    strategy_stats: Dict[str, Any] = {"strategy": "single_shot"}
    budget_warnings: List[str] = []
    extracted_chars_after = extracted_chars_before

    if use_map_reduce:
        combined_text = _flatten_file_context(file_context)
        chunk_chars = max(int(usable_chars * _MAP_CHUNK_BUDGET_SHARE), _MAP_REDUCE_MIN_CHARS // 2)
        try:
            answer_text, usage, finish_reason, mr_warnings, mr_stats = _execute_map_reduce(
                api_model=api_model,
                user_prompt=prompt,
                combined_text=combined_text,
                file_images=file_images or None,
                chunk_chars=chunk_chars,
                max_chunks=_MAP_REDUCE_MAX_CHUNKS,
            )
        except Exception as exc:
            return ExecutionResult(
                status="failed",
                error={"code": "provider_error", "message": f"{type(exc).__name__}: {exc}"},
                warnings=[f"Provider call failed for routed model '{selected_model}'."] + file_warnings,
            )
        strategy_stats = mr_stats
        strategy_stats["chunk_chars"] = chunk_chars
        strategy_stats["chunk_tokens_approx"] = chunk_chars // _CHARS_PER_TOKEN
        if not answer_text:
            return ExecutionResult(
                status="failed",
                error={"code": "map_reduce_failed", "message": "Map-reduce produced no answer."},
                warnings=file_warnings + mr_warnings,
            )
        extracted_chars_after = strategy_stats.get("chars_processed", extracted_chars_before)
        if mr_warnings:
            file_warnings.extend(f"Map-reduce: {w}" for w in mr_warnings)
    else:
        if file_context:
            file_context, budget_warnings = _enforce_context_budget(file_context, usable_chars)
        extracted_chars_after = sum(len(item.get("text", "")) for item in file_context)
        execution_prompt = prompt
        if file_paths:
            execution_prompt = _compose_execution_prompt(prompt=prompt, file_context=file_context)
        try:
            answer_text, usage, finish_reason = _call_openai(
                api_model=api_model,
                prompt=execution_prompt,
                images=file_images or None,
            )
        except Exception as exc:
            return ExecutionResult(
                status="failed",
                error={"code": "provider_error", "message": f"{type(exc).__name__}: {exc}"},
                warnings=[f"Provider call failed for routed model '{selected_model}'."] + file_warnings,
            )
        if budget_warnings:
            file_warnings.extend(f"Context budget: {w}" for w in budget_warnings)

    metadata: Dict[str, Any] = {
        "provider": "OpenAI",
        "routed_model": selected_model,
        "api_model": api_model,
        "finish_reason": finish_reason,
        "latency_ms": round((time.time() - start) * 1000, 2),
        "usage": usage,
        "files_used": file_paths,
        "input_formats": normalized_formats,
        "execution_strategy": strategy_stats.get("strategy", "single_shot"),
    }
    if file_paths:
        metadata["supports_multimodal_inputs"] = True
        metadata["content_extraction"] = {
            "chars_extracted": extracted_chars_before,
            "chars_sent_to_model": extracted_chars_after,
            "chars_truncated": max(0, extracted_chars_before - extracted_chars_after) if strategy_stats.get("strategy") == "single_shot" else 0,
            "model_context_window_tokens": budget["model_context_window_tokens"],
            "usable_context_tokens": budget["usable_context_tokens"],
            "prompt_tokens_est": budget["prompt_tokens_est"],
        }
    if strategy_stats.get("strategy") == "map_reduce":
        metadata["map_reduce"] = {k: v for k, v in strategy_stats.items() if k != "strategy"}
    if file_images:
        metadata["images_attached"] = [img["name"] for img in file_images]

    return ExecutionResult(
        status="completed",
        answer_text=answer_text,
        metadata=metadata,
        warnings=file_warnings,
    )


def _compose_execution_prompt(*, prompt: str, file_context: List[Dict[str, str]]) -> str:
    if not file_context:
        return prompt

    snippets = []
    for idx, item in enumerate(file_context, start=1):
        source = item.get("source", "file")
        name = item.get("name", f"file-{idx}")
        text = item.get("text", "").strip()
        if not text:
            continue
        snippets.append(f"[{idx}] {name} ({source}):\n{text}")

    if not snippets:
        return prompt

    joined = "\n\n".join(snippets)
    return (
        f"{prompt}\n\n"
        "Use the uploaded files as source context and answer from that context only:\n"
        f"{joined}"
    )


def _extract_file_contexts(
    file_paths: List[str],
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    contexts: List[Dict[str, str]] = []
    images: List[Dict[str, str]] = []
    warnings: List[str] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        if not path.exists():
            warnings.append(f"{path}: file not found")
            continue
        if not path.is_file():
            warnings.append(f"{path}: not a regular file")
            continue

        ext = path.suffix.lower()
        if ext == ".pdf":
            text, pdf_stats = _extract_pdf_text(str(path))
            if text:
                pages_read = pdf_stats.get("pages_read", 0)
                pages_total = pdf_stats.get("pages_total", 0)
                if pages_total and pages_read < pages_total:
                    warnings.append(
                        f"{path.name}: {pages_read} of {pages_total} PDF pages readable"
                    )
                contexts.append({
                    "name": path.name,
                    "source": f"pdf ({pages_read}/{pages_total} pages)" if pages_total else "pdf",
                    "text": _truncate(text, _PDF_TEXT_CHAR_LIMIT),
                })
            else:
                warnings.append(f"{path.name}: PDF text extraction failed")
            continue

        if ext in _EXECUTABLE_TEXT_EXTENSIONS:
            text = _read_text_file(str(path))
            if text:
                contexts.append(
                    {"name": path.name, "source": "text", "text": _truncate(text, _TEXT_FILE_CHAR_LIMIT)}
                )
            else:
                warnings.append(f"{path.name}: text extraction failed")
            continue

        if ext in _SUPPORTED_AUDIO_EXTENSIONS:
            if path.stat().st_size > _AUDIO_TRANSCRIPTION_MAX_BYTES:
                warnings.append(f"{path.name}: audio file too large for inline transcription")
                continue
            text = _transcribe_audio(str(path))
            if text:
                contexts.append(
                    {"name": path.name, "source": "audio", "text": _truncate(text, _TEXT_CONTEXT_CHARS)}
                )
            else:
                warnings.append(f"{path.name}: audio transcription failed")
            continue

        if ext in _SUPPORTED_IMAGE_EXTENSIONS:
            if path.stat().st_size > _IMAGE_MAX_BYTES:
                warnings.append(f"{path.name}: image file too large for inline attachment")
                continue
            data_url = _encode_image_data_url(path)
            if data_url:
                images.append({"name": path.name, "source": "image", "data_url": data_url})
            else:
                warnings.append(f"{path.name}: image encoding failed")
            continue

        mime = mimetypes.guess_type(str(path))[0] or ""
        if mime.startswith("text/") and ext not in _SUPPORTED_EXTENSIONS:
            text = _read_text_file(str(path))
            if text:
                contexts.append(
                    {"name": path.name, "source": "text", "text": _truncate(text, _TEXT_FILE_CHAR_LIMIT)}
                )
            else:
                warnings.append(f"{path.name}: text extraction failed")
            continue

        warnings.append(f"{path.name}: unsupported attachment type for execution")

    return contexts, images, warnings


def _lookup_model_context_window(model_name: str) -> int:
    try:
        from app.models.registry_builder import MODEL_REGISTRY
    except Exception:
        return _DEFAULT_CONTEXT_WINDOW
    for m in MODEL_REGISTRY:
        if m.get("name") == model_name:
            ctx = m.get("context") or {}
            return int(ctx.get("window") or _DEFAULT_CONTEXT_WINDOW)
    return _DEFAULT_CONTEXT_WINDOW


def _compute_context_char_budget(model_name: str, prompt: str) -> Dict[str, int]:
    window = _lookup_model_context_window(model_name)
    prompt_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN + 1)
    available_tokens = window - _OUTPUT_RESERVE_TOKENS - prompt_tokens
    usable_tokens = max(_MIN_CONTEXT_TOKENS, int(available_tokens * _BUDGET_UTILIZATION))
    return {
        "model_context_window_tokens": window,
        "prompt_tokens_est": prompt_tokens,
        "usable_context_tokens": usable_tokens,
        "usable_context_chars": usable_tokens * _CHARS_PER_TOKEN,
    }


def _chunk_text_at_boundaries(text: str, chunk_chars: int) -> List[str]:
    """Split text into chunks near natural boundaries (paragraph → line → sentence → space)."""
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text] if text else []
    chunks: List[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            search_start = max(start + int(chunk_chars * 0.6), start)
            best = -1
            for sep in ("\n\n", "\n", ". ", " "):
                pos = text.rfind(sep, search_start, end)
                if pos > best:
                    best = pos + len(sep)
                    break
            if best > start:
                end = best
        chunks.append(text[start:end])
        start = end
    return chunks


def _flatten_file_context(file_context: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for idx, item in enumerate(file_context, start=1):
        name = item.get("name", f"file-{idx}")
        source = item.get("source", "file")
        text = (item.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"[{idx}] {name} ({source}):\n{text}")
    return "\n\n".join(parts)


def _execute_map_reduce(
    *,
    api_model: str,
    user_prompt: str,
    combined_text: str,
    file_images: Optional[List[Dict[str, str]]],
    chunk_chars: int,
    max_chunks: int,
) -> Tuple[str, Dict[str, int], Optional[str], List[str], Dict[str, Any]]:
    """
    Map-reduce over a large document.
      map:    per-chunk extraction guided by the user's prompt.
      reduce: one final call over the concatenated extracts (images attached here).
    Returns (answer_text, aggregated_usage, finish_reason, warnings, stats).
    """
    warnings: List[str] = []
    chunks = _chunk_text_at_boundaries(combined_text, chunk_chars)
    stats: Dict[str, Any] = {
        "strategy": "map_reduce",
        "chunks_planned": len(chunks),
        "chunks_processed": 0,
        "chars_processed": 0,
        "map_calls": 0,
        "reduce_calls": 0,
    }
    if not chunks:
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, None, warnings, stats

    if len(chunks) > max_chunks:
        dropped_chars = sum(len(c) for c in chunks[max_chunks:])
        warnings.append(
            f"Map-reduce produced {len(chunks)} chunks; capped at {max_chunks}. "
            f"Tail content omitted (~{dropped_chars} chars / ~{dropped_chars // _CHARS_PER_TOKEN} tokens)."
        )
        chunks = chunks[:max_chunks]

    aggregated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    intermediates: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        map_prompt = (
            f"You are processing PART {i} OF {len(chunks)} of a larger document.\n\n"
            f"USER REQUEST (applies to the whole document, not just this part):\n{user_prompt}\n\n"
            f"For THIS PART ONLY, extract every fact, name, date, number, quote, or claim relevant "
            f"to the user request. Preserve specifics. Be thorough — your output will be merged "
            f"with extracts from the other parts. Do not answer the user request yet; just extract.\n\n"
            f"--- BEGIN PART {i}/{len(chunks)} ---\n{chunk}\n--- END PART {i}/{len(chunks)} ---"
        )
        try:
            part_text, part_usage, _ = _call_openai(api_model=api_model, prompt=map_prompt)
        except Exception as exc:
            warnings.append(f"Map call {i}/{len(chunks)} failed: {type(exc).__name__}: {exc}")
            continue
        intermediates.append(f"[Part {i}/{len(chunks)} extract]\n{part_text}")
        for k in aggregated_usage:
            aggregated_usage[k] += int(part_usage.get(k) or 0)
        stats["map_calls"] += 1
        stats["chunks_processed"] += 1
        stats["chars_processed"] += len(chunk)

    if not intermediates:
        return "", aggregated_usage, None, warnings + ["All map calls failed; no intermediates to reduce."], stats

    reduce_prompt = (
        f"USER REQUEST:\n{user_prompt}\n\n"
        f"Below are per-part extracts from a large document split into {len(chunks)} sequential parts. "
        f"Merge them into ONE coherent, well-structured response that fully addresses the user request. "
        f"Deduplicate overlaps, preserve specific facts (names, dates, numbers, quotes), and keep the "
        f"final response focused on what the user actually asked for.\n\n"
        + "\n\n".join(intermediates)
    )
    try:
        final_text, final_usage, finish_reason = _call_openai(
            api_model=api_model, prompt=reduce_prompt, images=file_images
        )
    except Exception as exc:
        warnings.append(f"Reduce call failed: {type(exc).__name__}: {exc}")
        return "", aggregated_usage, None, warnings, stats

    for k in aggregated_usage:
        aggregated_usage[k] += int(final_usage.get(k) or 0)
    stats["reduce_calls"] = 1

    return final_text, aggregated_usage, finish_reason, warnings, stats


def _enforce_context_budget(
    file_context: List[Dict[str, str]],
    budget_chars: int,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Truncate cumulative extracted text to fit within budget_chars; emit warnings."""
    warnings: List[str] = []
    total_chars = sum(len(item.get("text", "")) for item in file_context)
    if total_chars <= budget_chars:
        return file_context, warnings

    result: List[Dict[str, str]] = []
    remaining = budget_chars
    for item in file_context:
        text = item.get("text", "")
        name = item.get("name", "file")
        original_len = len(text)
        if remaining <= 0:
            warnings.append(
                f"{name}: entire content omitted from prompt (routed model's context budget exhausted)"
            )
            continue
        if original_len > remaining:
            keep = max(0, remaining - 200)
            truncated = text[:keep].rstrip() + "\n...[truncated to fit routed model's context window]"
            dropped_chars = original_len - keep
            warnings.append(
                f"{name}: {dropped_chars} characters (~{dropped_chars // _CHARS_PER_TOKEN} tokens) omitted "
                f"to fit routed model's context window"
            )
            new_item = dict(item)
            new_item["text"] = truncated
            result.append(new_item)
            remaining = 0
        else:
            result.append(item)
            remaining -= original_len
    return result, warnings


def _encode_image_data_url(path: Path) -> Optional[str]:
    try:
        mime = mimetypes.guess_type(str(path))[0]
        if not mime:
            ext = path.suffix.lower().lstrip(".")
            mime = f"image/{ext or 'jpeg'}"
        with open(path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def _extract_pdf_text(path: str) -> Tuple[Optional[str], Dict[str, int]]:
    """Extract text from every page. Returns (text, {'pages_read', 'pages_total'})."""
    stats = {"pages_read": 0, "pages_total": 0}
    try:
        from app.core.artifact_inspection import fitz
    except Exception:
        fitz = None

    if fitz is None:
        return None, stats
    try:
        doc = fitz.open(path)
        texts: List[str] = []
        stats["pages_total"] = doc.page_count
        try:
            for page_idx in range(doc.page_count):
                try:
                    page = doc.load_page(page_idx)
                    page_text = page.get_text("text")
                    if page_text:
                        texts.append(page_text)
                    stats["pages_read"] += 1
                except Exception:
                    continue
            return _clean_text("".join(texts)), stats
        finally:
            try:
                doc.close()
            except Exception:
                pass
    except Exception:
        return None, stats


def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", errors="ignore") as fh:
            return fh.read(_TEXT_FILE_CHAR_LIMIT)
    except Exception:
        return None


def _transcribe_audio(path: str) -> Optional[str]:
    from openai import OpenAI

    client = OpenAI()
    with open(path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return _clean_text(getattr(transcript, "text", "") or "")


def _clean_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\n +", "\n", cleaned)
    return cleaned.strip()


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 200].rstrip() + "\n...[truncated]"


def _call_openai(
    *,
    api_model: str,
    prompt: str,
    images: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, Dict[str, Any], Optional[str]]:
    from openai import OpenAI

    client = OpenAI()
    if images:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img["data_url"]}})
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=api_model,
        messages=messages,
    )
    choice = completion.choices[0]
    message = choice.message
    text = (message.content or "").strip()

    usage = {}
    if getattr(completion, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
            "completion_tokens": getattr(completion.usage, "completion_tokens", None),
            "total_tokens": getattr(completion.usage, "total_tokens", None),
        }

    return text, usage, getattr(choice, "finish_reason", None)
