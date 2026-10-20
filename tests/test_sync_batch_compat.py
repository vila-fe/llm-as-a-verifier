from llm_verifier.async_adapters import AdapterResponse, AsyncVerifierAdapter
from llm_verifier.fine_grained_reward import score_directed_pairs


class ImmediateAdapter(AsyncVerifierAdapter):
    async def complete(self, prompt, model, top_logprobs=20, images=None):
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
        )

    async def aclose(self):
        return None


def test_sync_batch_wrapper_accepts_an_async_adapter():
    """Catches a compatibility regression for existing blocking batch callers."""
    scores = score_directed_pairs(
        ImmediateAdapter(),
        {"task": [
            {"problem": "problem", "trace": "trace zero", "reward": 0},
            {"problem": "problem", "trace": "trace one", "reward": 0},
        ]},
        {"task": [(0, 1)]},
        [{"id": "overall", "name": "Overall", "description": "Does it work?"}],
        "",
        n_reps=1,
        max_workers=1,
        cache_file=None,
        model="test-model",
        progress=False,
    )

    assert scores == {"overall|task|0,1|0": {"score_A": 1.0, "score_B": 0.0}}
