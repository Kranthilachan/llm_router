# ATLAS Router

**Adaptive Task and LLM Allocation System** — a production-oriented LLM
routing control plane, exposed as a FastAPI service. ATLAS decides *which
model or execution plan* should handle a request, taking into account task
requirements, governance policy, runtime health, cost, latency,
uncertainty, and SLAs. It does not generate responses itself — routing is
provider-agnostic and fully separated from model execution.

## How it's organized

```
atlas_router/
├── app/
│   ├── main.py                 # FastAPI app entrypoint (uvicorn target)
│   ├── config.py                # version stamps + confidence thresholds
│   ├── models/
│   │   ├── schemas.py            # internal dataclasses (TaskFeatures, ExecutionPlan, ...)
│   │   ├── catalog.py             # provider -> model name catalog
│   │   └── registry_builder.py     # synthesizes the canonical MODEL_REGISTRY
│   ├── core/
│   │   ├── artifact_inspection.py   # PDF/image/audio/video/xlsx/pptx inspection
│   │   ├── semantic_parser.py        # deterministic + heuristic task parsing
│   │   ├── feasibility.py             # hard constraint filtering
│   │   ├── policy.py                   # governance / policy engine
│   │   ├── scoring.py                   # Bayesian quality, Pareto, utility, confidence
│   │   ├── planning.py                   # single-model & multi-stage plan generation
│   │   ├── router.py                      # route() — the main pipeline
│   │   └── formatting.py                   # human-readable decision summaries
│   └── api/
│       ├── schemas.py                       # pydantic request/response models
│       └── routes.py                         # FastAPI endpoints
├── tests/
│   └── test_router.py                         # end-to-end scenario tests
├── requirements.txt
├── .env.example
└── README.md
```

This mirrors the routing pipeline described in the ATLAS design doc:

```
Request -> Artifact Inspection -> Semantic Parsing -> Task Representation
   -> Feasibility Filtering -> Policy Enforcement -> Bayesian Quality
   -> Pareto Reduction -> Utility Scoring -> Confidence Estimation
   -> Execution Plan -> Route / Escalate / Abstain
```

## Setup (VS Code / local)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional artifact-inspection libraries (`pymupdf`, `Pillow`, `openpyxl`,
`python-pptx`, `mutagen`) are in `requirements.txt` but the router degrades
gracefully if any are missing — it just falls back to prompt-keyword
heuristics for that modality. `ffprobe` (from ffmpeg) is used for
audio/video duration if present on the host `PATH`.

## Run the service

```bash
uvicorn app.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger docs.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Implement a concurrent web crawler in Python with rate limiting and structured JSON output.",
    "input_formats": ["text"],
    "estimated_tokens": 3000,
    "estimated_output_tokens": 4000,
    "profile_name": "balanced"
  }'
```

Response shape (abridged):

```json
{
  "abstain": false,
  "escalate_to_human": false,
  "selected_plan": {
    "plan_id": "...",
    "plan_type": "single_model",
    "selected_model": "GPT-5.4",
    "fallback_models": ["Claude-Sonnet-4.6", "Gemini-3-Pro"],
    "verifier_models": [],
    "expected_latency_ms": 2150.4,
    "expected_cost_usd": 0.0182,
    "confidence": 0.61
  },
  "decision_record": { "...": "full auditable trace" },
  "summary_text": "=== Atlas Router Decision === ..."
}
```

## Endpoints

| Method | Path             | Purpose |
|--------|------------------|---------|
| POST   | `/route`         | Run the full routing pipeline for a request |
| POST   | `/outcome`       | Feed an observed outcome back into a model's Bayesian priors |
| GET    | `/models`        | List the canonical registry (summary view) |
| GET    | `/models/{name}` | Full registry entry for one model |
| GET    | `/versions`      | Current version stamps for every subsystem |
| GET    | `/health`        | Liveness probe |

## The Model Registry

`app/models/registry_builder.py` synthesizes a full canonical registry
(modalities, capabilities, pricing, context window, static/dynamic ops
telemetry, Bayesian priors, benchmark/performance signals, domain
expertise, safety scores, and verifier fitness) for every model listed in
`app/models/catalog.py`, using deterministic per-model seeding so values
are stable across restarts.

**This is a synthetic starting point**, not real benchmark data. In
production, replace `build_registry()` / `MODEL_REGISTRY` with a
database-backed or config-driven registry hydrated from real telemetry
and evaluation pipelines — the rest of the router (feasibility, policy,
scoring, planning) only depends on the registry entry *shape*, not on
`registry_builder.py` itself.

## Plugging in a real semantic parser

By default, task understanding uses `fallback_structured_semantic_parse()`
(keyword heuristics) in `app/core/semantic_parser.py`. To use a real
structured-output LLM parser instead, pass an `llm_parser` callable into
`route()` — it must return a `StructuredSemanticParse` (or an equivalent
dict) and is validated via `validate_structured_parse()`. This isn't wired
into the HTTP layer by default; add it in `app/api/routes.py` where
`route()` is called.

## Testing

```bash
pytest tests/ -v
```

Tests mirror the original design-doc demonstration scenarios: a coding
task, a high-risk legal contract review, audio summarization, an
offensive-security policy denial, a budget-constrained support task,
long-context research with citations, and file-driven conflict detection.

## Design principles (from the spec)

1. **Hard constraints override soft preferences** — feasibility filtering
   runs before any scoring.
2. **Policy is independent of scoring** — governance never hides inside
   utility weights.
3. **Runtime conditions matter** — latency, availability, queue pressure,
   and incident status all feed into routing, not just benchmarks.
4. **Uncertainty is quantified** — Thompson Sampling estimates confidence
   rather than returning a bare point estimate.
5. **Derive signals, don't duplicate them** — everything traces back to
   the canonical registry.
6. **Every decision is reproducible** — each `RoutingDecision` carries a
   reproducibility hash plus per-subsystem version stamps.
7. **Multi-stage workflows get structured planning** — OCR, document QA,
   summarization, etc. can be routed to different specialist models within
   one plan.
