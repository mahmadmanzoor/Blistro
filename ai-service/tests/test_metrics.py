import pytest

from app.embeddings import FakeEmbedder
from evals import metrics
from evals.build_dataset import quote_on_page
from evals.run_eval import page_match, pct


class Judge:
    """Returns a fixed object for whichever schema is requested."""

    def __init__(self, **by_schema):
        self.by_schema = by_schema

    def with_structured_output(self, schema):
        outer = self

        class _Runner:
            async def ainvoke(self, _messages):
                return schema(**outer.by_schema[schema.__name__])

        return _Runner()


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        ([True, True, True], 1.0),
        ([False, False], 0.0),
        ([True, False, True], (1 / 1 + 2 / 3) / 2),
        ([False, True], 0.5),
    ],
)
def test_average_precision(verdicts, expected):
    assert metrics.average_precision(verdicts) == pytest.approx(expected)


async def test_context_precision_pads_short_verdict_lists():
    judge = Judge(ChunkVerdicts={"useful": [False, True]})
    score = await metrics.context_precision(judge, "q", "ref", ["a", "b", "c"])
    assert score == pytest.approx(0.5)


async def test_context_recall_and_faithfulness():
    judge = Judge(
        Attribution={"total_statements": 4, "attributed_statements": 3},
        Faithfulness={"total_claims": 5, "supported_claims": 5},
    )
    assert await metrics.context_recall(judge, "ref", ["chunk"]) == 0.75
    assert await metrics.faithfulness(judge, "answer", "context") == 1.0
    assert await metrics.faithfulness(judge, "answer", "  ") is None


async def test_response_relevancy():
    emb = FakeEmbedder(256)
    same = Judge(
        GeneratedQuestions={
            "questions": ["how often is the xr-200 valve inspected"],
            "noncommittal": False,
        }
    )
    score = await metrics.response_relevancy(
        same, emb, "how often is the xr-200 valve inspected", "every 90 days"
    )
    assert score == pytest.approx(1.0)
    evasive = Judge(GeneratedQuestions={"questions": ["x"], "noncommittal": True})
    assert await metrics.response_relevancy(evasive, emb, "q", "I don't know") == 0.0


def test_quote_check_and_page_match():
    page = "Deposits are due on the 15th day of the following month.\nOther text."
    assert quote_on_page("deposits are due on the 15th day of the following month", page)
    assert not quote_on_page("deposits are due monthly on the first", page)

    class Item:
        document_id, page = "d", 5

    assert page_match("d", 4, 6, Item)
    assert not page_match("d", 6, 7, Item) and not page_match("x", 5, 5, Item)
    assert pct([10, 20, 30, 40], 0.5) in (20, 30)


def test_langfuse_experiment_upserts_dataset_and_scores():
    from evals.build_dataset import Item
    from evals.run_eval import log_langfuse_experiment

    calls: dict = {"items": [], "scores": []}

    class FakeDataset:
        items = [{"input": {"id": "q1", "question": "Q?"}}]

        def run_experiment(self, *, name, run_name, task, evaluators, metadata):
            for it in self.items:
                out = task(item=it)
                calls["scores"] += evaluators[0](
                    input=it["input"], output=out, expected_output="A", metadata=None
                )
            calls["run"] = (name, run_name)

    class FakeClient:
        def create_dataset(self, **kw):
            raise RuntimeError("exists")

        def create_dataset_item(self, **kw):
            calls["items"].append(kw)

        def get_dataset(self, name):
            calls["dataset"] = name
            return FakeDataset()

        def flush(self):
            pass

    item = Item(
        id="q1",
        question="Q?",
        answer="A",
        quote="",
        kind="factual",
        document_id="d",
        document_name="d.pdf",
        page=1,
        split="dev",
    )
    row = {
        "id": "q1",
        "answer": "A [p. 1]",
        "correctness": 5,
        "citation_ok": True,
        "faithfulness": 1.0,
        "response_relevancy": 0.9,
        "context_precision": None,
        "context_recall": 1.0,
    }
    name = log_langfuse_experiment([item], [row], "run-1", "dev", client=FakeClient())
    assert name == calls["dataset"] == "blistro-rag-dev"
    assert calls["items"][0]["id"] == "blistro-rag-dev-q1"
    assert calls["run"] == ("blistro-rag", "run-1")
    names = {e["name"] if isinstance(e, dict) else e.name for e in calls["scores"]}
    assert names == {
        "correctness",
        "citation_ok",
        "faithfulness",
        "response_relevancy",
        "context_recall",
    }
