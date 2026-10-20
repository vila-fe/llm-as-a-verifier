import asyncio
import json

import pytest

from llm_verifier.async_adapters import AdapterResponse, AsyncVerifierAdapter
from llm_verifier.fine_grained_reward import (
    AsyncLazyClient,
    score_directed_pairs_async,
)


class BarrierAdapter(AsyncVerifierAdapter):
    """In-memory adapter that only releases once two coroutines enter it."""

    def __init__(self):
        self.entered = 0
        self.max_active = 0
        self._active = 0
        self._both_entered = asyncio.Event()

    async def complete(self, prompt, model, top_logprobs=20, images=None):
        self.entered += 1
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        if self.entered == 2:
            self._both_entered.set()
        try:
            await asyncio.wait_for(self._both_entered.wait(), timeout=0.2)
            return AdapterResponse(
                text="<score_A> A </score_A><score_B> T </score_B>",
                tokens=[
                    "<score_A>", " A", "</score_A>",
                    "<score_B>", " T", "</score_B>",
                ],
                position_logprobs=[
                    [("<score_A>", 0.0)], [(" A", 0.0)],
                    [("</score_A>", 0.0)], [("<score_B>", 0.0)],
                    [(" T", 0.0)], [("</score_B>", 0.0)],
                ],
                usage=(1, 0, 1, 0),
            )
        finally:
            self._active -= 1

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_async_batch_scores_concurrently_and_persists(tmp_path):
    """Catches a sequential/thread-backed async batch route or non-atomic cache use."""
    adapter = BarrierAdapter()
    cache_file = tmp_path / "scores.json"
    tasks = {
        "task": [
            {"problem": "problem", "trace": "trace zero", "reward": 0},
            {"problem": "problem", "trace": "trace one", "reward": 0},
        ]
    }
    criteria = [{"id": "overall", "name": "Overall", "description": "Does it work?"}]

    scores = await score_directed_pairs_async(
        AsyncLazyClient(adapter),
        tasks,
        {"task": [(0, 1), (1, 0)]},
        criteria,
        "",
        n_reps=1,
        max_workers=2,
        cache_file=cache_file,
        model="test-model",
        progress=False,
    )

    assert adapter.max_active == 2
    assert scores == {
        "overall|task|0,1|0": {"score_A": 1.0, "score_B": 0.0},
        "overall|task|1,0|0": {"score_A": 1.0, "score_B": 0.0},
    }
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "scores": scores,
    }
