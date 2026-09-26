from types import SimpleNamespace

from ragqa.chunking import Chunk
from ragqa.generator import NO_CONTEXT_ANSWER, ClaudeGenerator, build_prompt, extractive_answer

CONTEXTS = [
    Chunk("FedAvg averages client model updates.", "fl.pdf", 3, 0),
    Chunk("Docker builds container images.", "docker.md", None, 1),
]


class FakeClient:
    """Stands in for anthropic.Anthropic so tests need no API key or network."""

    def __init__(self, stop_reason="end_turn", text=""):
        self.calls = []
        response = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])

        def create(**kwargs):
            self.calls.append(kwargs)
            return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_prompt_numbers_passages_with_sources_and_pages():
    prompt = build_prompt("How are updates combined?", CONTEXTS)
    assert "[1] (fl.pdf, p. 3)" in prompt
    assert "[2] (docker.md)" in prompt
    assert prompt.endswith("Question: How are updates combined?")


def test_answer_returns_model_text():
    client = FakeClient(text="Updates are averaged with FedAvg [1].")
    answer = ClaudeGenerator(model="claude-opus-5", client=client).answer("How?", CONTEXTS)
    assert answer == "Updates are averaged with FedAvg [1]."
    assert client.calls[0]["model"] == "claude-opus-5"


def test_refusal_falls_back_to_best_passage():
    assert ClaudeGenerator(client=FakeClient(stop_reason="refusal")).answer("How?", CONTEXTS) == CONTEXTS[0].text


def test_no_contexts_skips_the_api_call():
    client = FakeClient()
    assert ClaudeGenerator(client=client).answer("How?", []) == NO_CONTEXT_ANSWER
    assert client.calls == []


def test_extractive_answer():
    assert extractive_answer(CONTEXTS) == CONTEXTS[0].text
    assert extractive_answer([]) == NO_CONTEXT_ANSWER
