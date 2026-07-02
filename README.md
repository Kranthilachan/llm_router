# ATLAS Router

**Adaptive Task and LLM Allocation System** — a FastAPI service that decides *which LLM should handle a request* and (optionally) *runs the request against that model and returns the answer*.

ATLAS separates two concerns that most LLM apps mix together:

1. **Routing** — given a prompt (and optional files), decide the best model based on task type, cost, latency, context window, governance policy, and runtime health.
2. **Execution** — once the model is picked, actually call it and return the answer. Supports PDFs, text files, images, and audio.

The router is provider-agnostic; execution currently uses OpenAI (`GPT-4o`, `GPT-4o-Mini`, `o3`) and Whisper.

---

## Contents

- [Quick start](#quick-start)
- [The two endpoints — which to use](#the-two-endpoints--which-to-use)
- [`POST /route` (JSON, routing-first)](#post-route-json-routing-first)
- [`POST /route/upload` (files + auto-execute)](#post-routeupload-files--auto-execute)
- [What files are supported](#what-files-are-supported)
- [Large documents (auto-chunking)](#large-documents-auto-chunking)
- [Other endpoints](#other-endpoints)
- [How routing works (pipeline)](#how-routing-works-pipeline)
- [Project structure](#project-structure)
- [Configuration (env vars)](#configuration-env-vars)
- [The model registry](#the-model-registry)
- [Design principles](#design-principles)
- [Development](#development)

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Required for executing routed answers via OpenAI / Whisper
export OPENAI_API_KEY=sk-...

uvicorn app.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger docs.

Sanity check:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

---

## The two endpoints — which to use

| Endpoint | Content-Type | `execute_answer` default | Use it when… |
|---|---|---|---|
| `POST /route` | JSON | **`false`** (opt in) | Text-only prompt. Testing routing logic. You want the decision report, not the answer. |
| `POST /route/upload` | JSON **or** multipart/form-data | **`true`** (opt out) | Uploading files (PDF/image/audio). You want the answer generated end-to-end. Also works for text-only prompts. |

Rule of thumb:
- **Want the answer?** Either endpoint works. `/route/upload` is easier because it defaults to executing.
- **Just want to inspect routing?** Use `/route`.
- **Have files?** Use `/route/upload`.

---

## `POST /route` (JSON, routing-first)

Accepts a JSON body. Returns the routing decision. By default it does **not** call the model — it just tells you which model would be picked and why. To also run the model and get an answer, set `"execute_answer": true`.

### Minimal example (routing only)

```bash
curl -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Suggest me the best family car for 2026"}'
```

Response gives you `selected_plan` (the routing decision) and `answer_text: null` (execution was not requested).

### With execution

```bash
curl -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Suggest me the best family car for 2026",
    "execute_answer": true
  }'
```

Response now includes `answer_text` populated by the routed model.

### Full request body (all fields optional except `prompt`)

```json
{
  "prompt": "Summarize the contract and highlight the payment clause",
  "input_formats": ["text"],
  "estimated_tokens": 2000,
  "estimated_output_tokens": 1200,
  "profile_name": "balanced",
  "execute_answer": false,
  "request_constraints": {
    "max_cost_usd": 0.05,
    "max_latency_ms": 5000,
    "min_confidence": 0.5
  },
  "tenant_context": {
    "tenant_id": "acme-corp",
    "allowed_models": ["GPT-4o", "Claude-Sonnet-4.6"]
  }
}
```

---

## `POST /route/upload` (files + auto-execute)

The user-facing endpoint. Send it a prompt and (optionally) one or more files, and it will route + execute the LLM and return the answer. Defaults are chosen so the common case works with the least configuration.

Accepts **two content types** on the same endpoint:

- `application/json` — same schema as `/route`. Convenient if your client only sends JSON.
- `multipart/form-data` — required if you want to upload files.

`execute_answer` defaults to `true`. Send `false` if you only want routing.

### With a file (multipart form-data)

```bash
curl -X POST http://127.0.0.1:8000/route/upload \
  -F "prompt=Summarize this document in 5 bullets and highlight key risks" \
  -F "files=@/path/to/contract.pdf"
```

Form fields (all optional except `prompt`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | text | — | Required |
| `files` | file | — | Upload as many as you like. Use `files` (plural) even for a single file. Multiple files → repeat the key. |
| `input_formats` | text | auto-inferred from file extensions | Only override if you want to signal a modality the extension doesn't imply |
| `estimated_tokens` | int | 2000 | Prompt+context token estimate. Affects feasibility filtering. |
| `estimated_output_tokens` | int | 1200 | Expected output size. Affects SLA checks. |
| `profile_name` | text | `balanced` | Routing profile (see `app/core/scoring.py`) |
| `execute_answer` | bool | `true` | Send `false` for routing-only |
| `artifact_hints` | JSON array | none | Optional overrides, e.g. `[{"format":"pdf","page_count":45}]` |
| `shadow_model` | text | none | Compare against a shadow model |

### Multiple files at once

```bash
curl -X POST http://127.0.0.1:8000/route/upload \
  -F "prompt=Summarize the PDF and describe the image" \
  -F "files=@/path/to/report.pdf" \
  -F "files=@/path/to/diagram.png"
```

Use the same field key `files` for each file — FastAPI collects them into a list.

### With JSON body (no file upload)

```bash
curl -X POST http://127.0.0.1:8000/route/upload \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Suggest me the best family car for 2026"}'
```

Same shape as `/route`, but `execute_answer` defaults to `true`.

### Response shape

```jsonc
{
  "abstain": false,
  "escalate_to_human": false,
  "selected_plan": {
    "plan_id": "0d80ccc6-e7d",
    "plan_type": "single_model",
    "selected_model": "GPT-4o",
    "fallback_models": ["o3", "GPT-4o-Mini"],
    "expected_latency_ms": 1854.4,
    "expected_cost_usd": 0.02368,
    "expected_quality": 0.759,
    "confidence": 0.61,
    "explanation": { "primary_model": "GPT-4o", "task_family": "document_qa", "...": "..." },
    "trace": { "router_version": "atlas-router-v1.0.0", "...": "..." }
  },
  "decision_record": { "...": "full auditable trace" },
  "summary_text": "=== Atlas Router Decision === ...",

  // Execution output (present when execute_answer=true)
  "answer_text": "Here is your summary...",
  "execution_status": "completed",
  "execution_metadata": {
    "provider": "OpenAI",
    "routed_model": "GPT-4o",
    "api_model": "gpt-4o",
    "latency_ms": 6821.5,
    "usage": { "prompt_tokens": 1542, "completion_tokens": 125, "total_tokens": 1667 },
    "files_used": ["/tmp/atlas-route-upload-xyz/tmpXXXX.pdf"],
    "input_formats": ["pdf"],
    "execution_strategy": "single_shot",
    "content_extraction": {
      "chars_extracted": 4567,
      "chars_sent_to_model": 4567,
      "chars_truncated": 0,
      "model_context_window_tokens": 125927,
      "usable_context_tokens": 99114
    }
  },
  "execution_warnings": [],
  "execution_error": null
}
```

Possible `execution_status` values:
- `"completed"` — model responded successfully.
- `"unsupported"` — plan or content type not executable in v1 (see warnings).
- `"escalated"` — the routing decision escalated to human; execution skipped.
- `"abstained"` — router chose to abstain; execution skipped.
- `"failed"` — provider call failed; see `execution_error`.

---

## What files are supported

| Format | Extension | How it's processed | Cap |
|---|---|---|---|
| PDF | `.pdf` | Text extracted from **every page** via PyMuPDF | Read fully; capped by model's context window |
| Plain text | `.txt`, `.md` | Read as-is | 5 MB safety cap; capped by model window |
| Image | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | Base64-encoded and sent as multimodal `image_url` part | 20 MB per image |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac` | Transcribed via OpenAI Whisper, then passed as text context | 24 MB per file (Whisper limit) |

**Not supported yet:** video (`.mp4`, `.mov`, etc.) — would require ffmpeg-based audio+keyframe extraction. Excel/PowerPoint are read for routing (artifact inspection) but not sent to execution.

If you upload a file with `input_formats` unset, the server auto-infers the modality from the file extension.

---

## Large documents (auto-chunking)

If an uploaded document exceeds the routed model's context window, the execution layer **automatically switches to map-reduce** instead of silently truncating:

1. Split the extracted text into chunks (default ~55% of the usable context window per chunk).
2. **Map:** call the routed model once per chunk to extract facts relevant to the user's prompt.
3. **Reduce:** call the routed model one final time over the concatenated per-chunk extracts, producing the final answer.

You'll see this in `execution_metadata`:

```jsonc
"execution_strategy": "map_reduce",
"map_reduce": {
  "chunks_planned": 10,
  "chunks_processed": 10,
  "chars_processed": 1971071,
  "map_calls": 10,
  "reduce_calls": 1,
  "chunk_chars": 218050
}
```

Constraints:
- Max 20 chunks (~10 MB of extracted text). Anything beyond that is truncated with a warning.
- All chunks use the same routed model.
- Images are attached to the reduce call only.
- Map calls are currently sequential — expect longer wall-clock time for very large docs.

If the document fits in the model's window, single-shot execution is used (no chunking).

---

## Other endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/outcome` | Report an observed outcome (success, quality, latency, cost) so Bayesian priors update |
| GET | `/models` | List all models in the canonical registry (summary view) |
| GET | `/models/{name}` | Full registry entry for one model |
| GET | `/versions` | Version stamps for router, parser, policy, scoring, registry, calibration |
| GET | `/health` | Liveness probe |

### Reporting an outcome

```bash
curl -X POST http://127.0.0.1:8000/outcome \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "GPT-4o",
    "task_family": "document_qa",
    "success": true,
    "quality_score": 0.85,
    "latency_ms": 2400,
    "cost_usd": 0.02,
    "user_accepted": true,
    "safety_flagged": false
  }'
```

This updates the Beta-distribution priors used in future routing decisions for that model + task family.

---

## How routing works (pipeline)

```
Request
   -> Artifact Inspection      (PDF pages, image dims, audio duration...)
   -> Semantic Parsing          (task family, domain, risk tier, workflow)
   -> Task Representation       (canonical TaskFeatures)
   -> Feasibility Filtering     (modality, capability, context window, budget)
   -> Policy Enforcement        (medical/legal/finance/security gates)
   -> Bayesian Quality          (contextual quality per candidate)
   -> Pareto Reduction          (drop dominated candidates)
   -> Utility Scoring           (profile-weighted cost/latency/quality)
   -> Confidence Estimation     (Thompson Sampling)
   -> Execution Plan            (single-model or multi-stage)
   -> Route / Escalate / Abstain
```

Every step is traced in `decision_record.pipeline_trace` and every subsystem's version is stamped for reproducibility.

---

## Project structure

```
atlas_router/
├── app/
│   ├── main.py                        # FastAPI app entrypoint
│   ├── config.py                       # version stamps + env-var thresholds
│   ├── models/
│   │   ├── schemas.py                   # internal dataclasses
│   │   ├── catalog.py                    # provider -> model name catalog (~95 models)
│   │   └── registry_builder.py            # builds the canonical MODEL_REGISTRY
│   ├── core/
│   │   ├── artifact_inspection.py          # PDF/image/audio/xlsx/pptx probing
│   │   ├── semantic_parser.py               # deterministic + heuristic task parsing
│   │   ├── domain_classifier.py              # fastembed embeddings for domain classification
│   │   ├── llm_domain_parser.py               # optional GPT-4o-Mini tiebreaker
│   │   ├── feasibility.py                      # hard constraint filtering
│   │   ├── policy.py                            # governance / policy engine
│   │   ├── scoring.py                            # Bayesian quality, Pareto, utility
│   │   ├── planning.py                            # single-model + multi-stage planning
│   │   ├── router.py                               # route() — the main pipeline
│   │   ├── formatting.py                            # human-readable decision summaries
│   │   └── execution.py                              # optional LLM call (OpenAI/Whisper)
│   └── api/
│       ├── schemas.py                                  # pydantic request/response models
│       └── routes.py                                    # FastAPI endpoints
├── data/
│   └── domain_examples.json                              # labeled prompts for domain classifier
├── tests/
│   └── test_router.py                                     # end-to-end scenario tests
├── scripts/
│   └── demo.py                                             # local demo script
├── requirements.txt
└── README.md
```

---

## Configuration (env vars)

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for `execute_answer=true`, Whisper transcription, and the LLM domain-parser tiebreaker |
| `ATLAS_ALLOWED_PROVIDERS` | none | Comma-separated list, e.g. `OpenAI,Anthropic`. Restricts routing to these providers by default. Per-request `allowed_providers` always wins. |
| `ATLAS_ABSTAIN_THRESHOLD` | `0.40` | Confidence below this → abstain |
| `ATLAS_ESCALATE_THRESHOLD` | `0.55` | Confidence below this → escalate to human |
| `ATLAS_HIGH_THRESHOLD` | `0.75` | High-risk tasks below this → escalate |
| `ATLAS_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 384-dim embedding model for domain classifier |
| `ATLAS_LLM_PARSER_MODEL` | `gpt-4o-mini` | Model used by the optional LLM tiebreaker |

An example file lives at `.env.example`.

---

## The model registry

`app/models/registry_builder.py` synthesizes a full canonical registry (modalities, capabilities, pricing, context window, static/dynamic ops telemetry, Bayesian priors, benchmark signals, domain expertise, safety scores, verifier fitness) for every model listed in `app/models/catalog.py`, using deterministic per-model seeding so values are stable across restarts.

**This is a synthetic starting point**, not real benchmark data. In production, replace `MODEL_REGISTRY` with a database-backed or config-driven registry hydrated from real telemetry and evaluation pipelines. The rest of the router (feasibility, policy, scoring, planning) only depends on the registry entry *shape*, not on `registry_builder.py` itself.

The domain classifier at `app/core/domain_classifier.py` reads labeled prompts from `data/domain_examples.json`. Currently covered domains: `crm`, `hrm`, `project`, `accounts`. Add prompts to any bucket and restart the server — no code changes needed.

---

## Design principles

1. **Hard constraints override soft preferences** — feasibility filtering runs before any scoring.
2. **Policy is independent of scoring** — governance never hides inside utility weights.
3. **Runtime conditions matter** — latency, availability, queue pressure, and incident status feed into routing, not just benchmarks.
4. **Uncertainty is quantified** — Thompson Sampling estimates confidence rather than returning a bare point estimate.
5. **Derive signals, don't duplicate them** — everything traces back to the canonical registry.
6. **Every decision is reproducible** — each `RoutingDecision` carries a reproducibility hash plus per-subsystem version stamps.
7. **Multi-stage workflows get structured planning** — OCR, document QA, summarization can be routed to different specialist models within one plan.
8. **Routing is separate from execution** — routing decides which model; execution is optional and swappable per provider.

---

## Development

```bash
# Run locally
uvicorn app.main:app --reload --port 8000

# Interactive API docs
open http://127.0.0.1:8000/docs
```

### Plugging in a real LLM semantic parser

By default, task understanding uses `fallback_structured_semantic_parse()` (keyword heuristics) in `app/core/semantic_parser.py`. To use a real structured-output LLM parser instead, pass an `llm_parser` callable into `route()` — it must return a `StructuredSemanticParse`. The bundled `hybrid_tiebreaker_parser` in `app/core/llm_domain_parser.py` is one such implementation.

### Optional libraries

Artifact-inspection libraries (`pymupdf`, `Pillow`, `openpyxl`, `python-pptx`, `mutagen`) are in `requirements.txt` but the router degrades gracefully if any are missing — it just falls back to prompt-keyword heuristics for that modality. `ffprobe` (from ffmpeg) is used for audio/video duration if present on `PATH`.

### Common gotchas

- **"Field required: prompt" on `/route/upload`** — you probably sent JSON to the form-data endpoint. Either send it as `multipart/form-data`, or use JSON with the correct `Content-Type: application/json` header (both work now).
- **`answer_text: null` on `/route`** — you didn't set `"execute_answer": true`. `/route` is routing-first; add the flag to also execute.
- **`execution_status: "unsupported"` with an image or audio file** — check `input_formats` isn't accidentally set to something the extractor doesn't understand. Auto-inference handles the common cases if you just leave it off.
- **Model context window overflowed** — check `execution_metadata.execution_strategy`. If `map_reduce`, the doc was chunked automatically. If `single_shot` and `chars_truncated > 0`, some content was dropped (with warnings in `execution_warnings`).
