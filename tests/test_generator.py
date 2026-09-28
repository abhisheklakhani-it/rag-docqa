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

        client = self

        class Stream:
            """Mimics the SDK's streaming context manager: text pieces, then the final message."""

            def __init__(self, **kwargs):
                client.calls.append(kwargs)
                self.text_stream = iter(text.split("|"))  # "|" separates the streamed pieces

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create, stream=Stream))


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


def test_stream_yields_pieces_in_order():
    client = FakeClient(text="Updates are |averaged |[1].")
    pieces = list(ClaudeGenerator(client=client).stream("How?", CONTEXTS))
    assert pieces == ["Updates are ", "averaged ", "[1]."]
    assert client.calls[0]["messages"][0]["content"] == build_prompt("How?", CONTEXTS)


def test_stream_refusal_adds_best_passage():
    pieces = list(ClaudeGenerator(client=FakeClient(stop_reason="refusal", text="I can't")).stream("How?", CONTEXTS))
    assert "declined" in pieces[-1] and CONTEXTS[0].text in pieces[-1]


def test_stream_without_contexts_skips_the_api_call():
    client = FakeClient()
    assert list(ClaudeGenerator(client=client).stream("How?", [])) == [NO_CONTEXT_ANSWER]
    assert client.calls == []
