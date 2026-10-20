# Async LLM Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move verifier HTTP requests to explicit `httpx.AsyncClient` adapters while retaining callable synchronous entry points and response-score compatibility.

**Architecture:** Add a focused `async_adapters.py` boundary that normalizes OpenAI-compatible, DeepSeek, and Vertex Gemini REST responses into a common async result. Add an atomic, versioned JSON cache helper, then have a concurrent coroutine batch scorer use `asyncio` tasks and a semaphore; retain the current synchronous APIs as compatibility wrappers or legacy-client paths.

**Tech Stack:** Python 3.9+, `httpx.AsyncClient`, `asyncio`, `pytest`, `pytest-asyncio`.

**Spec:** User task: refactor `llm_verifier/fine_grained_reward.py` to async `httpx` adapters, add offline contract/cache tests, atomic JSON persistence, dependencies, branch, and local commits.

## Global Constraints

- Work only in `/home/ubuntu/repo-audit-10-llm-as-a-verifier`.
- Use `httpx.AsyncClient` for adapter HTTP; do not use `ThreadPoolExecutor` in async code paths.
- Preserve `(text, tokens, position_logprobs)` and `extract_score` behavior.
- Cache writes use a temporary file, `flush`, `fsync`, and `os.replace`; use schema version 1 and read legacy plain-dict cache files.
- Tests must make no network requests; adapter transport is `httpx.MockTransport`.
- Keep synchronous compatibility; do not push.

---

### Task 1: Adapter and cache contract tests

**Files:**
- Create: `tests/test_async_adapters.py`
- Create: `tests/test_score_cache.py`

**Interfaces:**
- Consumes: `httpx.MockTransport`.
- Produces: behavioral expectations for `AsyncVerifierAdapter.complete`, `call_verifier_async`, `load_score_cache`, and `save_score_cache`.

- [ ] **Step 1: Write failing adapter contract tests**

```python
adapter = OpenAICompatibleAsyncAdapter(
    base_url="https://verifier.test/v1", api_key="test", client=client
)
response = await adapter.complete("rate this", model="test-model")
assert response.text == "analysis <score_A> A </score_A>"
assert response.position_logprobs == [[(" A", -0.1), (" B", -2.0)]]
```

- [ ] **Step 2: Run the adapter test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_async_adapters.py -q`
Expected: import failure because `llm_verifier.async_adapters` does not yet exist.

- [ ] **Step 3: Write failing atomic-cache tests**

```python
save_score_cache(cache_path, {"criterion|task|0,1|0": {"score_A": 1.0, "score_B": 0.0}})
assert json.loads(cache_path.read_text())["schema_version"] == 1
assert load_score_cache(cache_path)["criterion|task|0,1|0"]["score_A"] == 1.0
```

- [ ] **Step 4: Run the cache test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_score_cache.py -q`
Expected: import failure because `llm_verifier.score_cache` does not yet exist.

### Task 2: Async adapter and atomic cache implementation

**Files:**
- Create: `llm_verifier/async_adapters.py`
- Create: `llm_verifier/score_cache.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `AsyncVerifierAdapter.complete(prompt, model, top_logprobs, images) -> AdapterResponse` and `load_score_cache(path) -> dict`, `save_score_cache(path, scores) -> None`.

- [ ] **Step 1: Implement the three adapter payload/response mappings**

```python
class AsyncVerifierAdapter(ABC):
    @abstractmethod
    async def complete(self, prompt, model, top_logprobs=20, images=None):
        raise NotImplementedError
```

OpenAI-compatible requests post to `chat/completions`, DeepSeek extends that request with its thinking fields, and Gemini posts `generateContent` requests to the Vertex REST endpoint. Every call uses an injected or owned `httpx.AsyncClient` and parses provider JSON into one `AdapterResponse`.

- [ ] **Step 2: Implement atomic, schema-v1 cache persistence**

```python
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump({"schema_version": CACHE_SCHEMA_VERSION, "scores": scores}, handle)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary_path, cache_path)
```

On any exception, remove the temporary file; accept old plain score mappings when loading.

- [ ] **Step 3: Declare dependencies**

```toml
dependencies = ["httpx>=0.27", ...]
[project.optional-dependencies]
test = ["pytest>=8", "pytest-asyncio>=0.23"]
```

- [ ] **Step 4: Run the contract suite to verify green**

Run: `.venv/bin/python -m pytest tests/test_async_adapters.py tests/test_score_cache.py -q`
Expected: all tests pass without network access.

### Task 3: Async dispatch and compatible batch scoring

**Files:**
- Modify: `llm_verifier/fine_grained_reward.py`
- Modify: `tests/test_async_adapters.py`

**Interfaces:**
- Produces: `create_async_client`, `call_verifier_async`, `score_pair_criterion_async`, `score_directed_pairs_async`.
- Preserves: `call_verifier`, `score_pair_criterion`, `score_directed_pairs` return shapes.

- [ ] **Step 1: Write a failing batch concurrency/format test**

```python
scores = await score_directed_pairs_async(..., max_workers=2, cache_file=cache)
assert scores[key] == {"score_A": 1.0, "score_B": 0.0}
assert adapter.max_active >= 2
```

The adapter’s in-memory barrier demonstrates coroutine concurrency without an HTTP request and catches a sequential or thread-pool async implementation.

- [ ] **Step 2: Run the batch test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_async_adapters.py::test_async_batch_scores_concurrently_and_persists -q`
Expected: missing `score_directed_pairs_async`.

- [ ] **Step 3: Implement async dispatch and two-phase batch scoring**

Use `asyncio.create_task`, `asyncio.as_completed`, and an `asyncio.Semaphore`; preserve warm-prefix then remaining work, swap scores for odd repetitions, ties-on-error, progress, and periodic atomic cache persistence. The synchronous scorer uses the async batch when it owns an async adapter and retains a legacy path only for explicitly supplied SDK clients.

- [ ] **Step 4: Run all tests and static syntax validation**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m compileall -q llm_verifier`
Expected: pass.

### Task 4: Final review and commits

**Files:**
- Review all changed files.

- [ ] **Step 1: Verify package metadata and git diff**

Run: `git diff --check && git status --short && git diff --cached`
Expected: no whitespace error or unexpected file.

- [ ] **Step 2: Create atomic local commits**

```bash
git add pyproject.toml requirements.txt tests llm_verifier/async_adapters.py llm_verifier/score_cache.py
git commit -m "feat: add async verifier adapters and atomic cache"
git add llm_verifier/fine_grained_reward.py tests
git commit -m "refactor: route batch scoring through async adapters"
```

- [ ] **Step 3: Record exact verification and branch state**

Run: `git log --oneline origin/main..HEAD && git status --short --branch`
Expected: local `refactor/async-llm-adapters` ahead of `main`, clean, not pushed.
