"""Run sgl-eval while preserving OpenAI reasoning-channel text.

Some reasoning models return all generated text in ``reasoning_content`` and
leave ``message.content`` empty.  sgl-eval 0.0.1 currently reads only
``message.content``, which turns a successful generation into an empty answer.
This entry point keeps the installed harness intact and changes only its
response-to-Sample adapter so mathematical graders see both response channels.
"""

from __future__ import annotations

from typing import Any, Optional


def _message_text(message: Any) -> str:
    reasoning = getattr(message, "reasoning_content", None) or ""
    content = getattr(message, "content", None) or ""
    return "\n".join(part for part in (reasoning, content) if part)


def _install_reasoning_compat() -> None:
    from sgl_eval.sampler import ChatCompletionSampler

    original = ChatCompletionSampler._to_sample

    def _to_sample(
        response: Any,
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ):
        sample = original(response, start=start, end=end)
        text = _message_text(response.choices[0].message)
        if text:
            sample.text = text
        return sample

    ChatCompletionSampler._to_sample = staticmethod(_to_sample)


def main() -> int:
    _install_reasoning_compat()
    from sgl_eval.cli import main as sgl_eval_main

    return sgl_eval_main()


if __name__ == "__main__":
    raise SystemExit(main())
