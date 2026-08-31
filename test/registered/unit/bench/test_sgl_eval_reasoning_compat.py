from types import SimpleNamespace

from sglang.test.sgl_eval_reasoning_compat import _message_text


def test_message_text_combines_reasoning_before_final_content():
    message = SimpleNamespace(
        reasoning_content="work through the problem",
        content="\\boxed{42}",
    )

    assert _message_text(message) == "work through the problem\n\\boxed{42}"


def test_message_text_accepts_reasoning_only_response():
    message = SimpleNamespace(reasoning_content="answer is 42", content="")

    assert _message_text(message) == "answer is 42"
