"""Turn retrieved chunks into an answer: with Claude when configured, extractively otherwise."""

import logging
from collections.abc import Iterator

import anthropic

from .chunking import Chunk

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "No relevant passages were found in the indexed documents."

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite the passages you use inline, like [1] or [2]. "
    "If the passages do not contain the answer, say you don't know instead of guessing."
)


def format_source(chunk: Chunk) -> str:
    return f"{chunk.source}, p. {chunk.page}" if chunk.page else chunk.source


def build_prompt(question: str, contexts: list[Chunk]) -> str:
    passages = "\n\n".join(f"[{i}] ({format_source(c)})\n{c.text}" for i, c in enumerate(contexts, 1))
    return f"Context passages:\n\n{passages}\n\nQuestion: {question}"


def extractive_answer(contexts: list[Chunk]) -> str:
    """Fallback without an LLM: return the best-matching passage as-is."""
    return contexts[0].text if contexts else NO_CONTEXT_ANSWER


class ClaudeGenerator:
    def __init__(self, model: str = "claude-opus-5", client: anthropic.Anthropic | None = None) -> None:
        self.model = model
        self.client = client or anthropic.Anthropic()

    def _request(self, question: str, contexts: list[Chunk]) -> dict:
        return {
            "model": self.model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            # If the model declines a request, the API retries it on a fallback model.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
            "messages": [{"role": "user", "content": build_prompt(question, contexts)}],
        }

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return NO_CONTEXT_ANSWER
        response = self.client.beta.messages.create(**self._request(question, contexts))
        if response.stop_reason == "refusal":
            logger.warning("Model declined to answer; returning extractive answer")
            return extractive_answer(contexts)
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def stream(self, question: str, contexts: list[Chunk]) -> Iterator[str]:
        """Yield the answer piece by piece as the model writes it."""
        if not contexts:
            yield NO_CONTEXT_ANSWER
            return
        with self.client.beta.messages.stream(**self._request(question, contexts)) as stream:
            yield from stream.text_stream
            if stream.get_final_message().stop_reason == "refusal":
                # Text already sent can't be taken back, so add a clear note instead.
                logger.warning("Model declined to answer during streaming")
                yield "\n\n(The model declined to answer. Best matching passage:)\n\n" + extractive_answer(contexts)
