"""Turn retrieved chunks into an answer: with Claude when configured, extractively otherwise."""

import logging

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

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return NO_CONTEXT_ANSWER
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            # If the model declines a request, the API retries it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        )
        if response.stop_reason == "refusal":
            logger.warning("Model declined to answer; returning extractive answer")
            return extractive_answer(contexts)
        return "".join(block.text for block in response.content if block.type == "text").strip()
