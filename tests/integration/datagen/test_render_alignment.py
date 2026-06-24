"""Guard for the multimodal chat-path ``add_generation_prompt=False`` invariant.

For multimodal data, speculators sends raw ``messages`` to vLLM's Chat
Completions API, which re-tokenizes them server-side.  Preprocessing builds
``input_ids`` with ``add_generation_prompt=False`` (``preprocessing.py:404``),
but vLLM defaults that flag to ``True``.  The explicit pin at
``vllm_client.py:203`` (async) / ``:245`` (sync) is the only thing keeping
the served tokens aligned with the training tokens.

This test drives the real request builders against an in-process httpx
transport and asserts the wire bytes.  No model, no GPU, no vllm needed.
"""

import json

import httpx
import openai
import pytest

from speculators.data_generation.vllm_client import (
    generate_hidden_states,
)

MM_MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "Describe this."}]}
]
TOKEN_IDS = [1, 2, 3]
HS_PATH = "/tmp/hs.pt"


def _vllm_like_response(request: httpx.Request) -> httpx.Response:
    if "/chat/" in str(request.url):
        payload = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": ""},
                }
            ],
            "prompt_token_ids": TOKEN_IDS,
            "kv_transfer_params": {"hidden_states_path": HS_PATH},
        }
    else:
        payload = {
            "id": "x",
            "object": "text_completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "text": "",
                    "prompt_token_ids": TOKEN_IDS,
                }
            ],
            "kv_transfer_params": {"hidden_states_path": HS_PATH},
        }
    return httpx.Response(200, json=payload)


def _capturing_client(captured: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"url": str(request.url), "body": json.loads(request.content)})
        return _vllm_like_response(request)

    return openai.Client(
        api_key="x",
        base_url="http://test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.regression
def test_chat_path_pins_add_generation_prompt_false():
    """Chat request pins ``add_generation_prompt=False``; completions path
    serves ``input_ids`` verbatim without it."""
    captured: list[dict] = []
    client = _capturing_client(captured)

    # multimodal chat path
    out = generate_hidden_states(
        client, "m", {"input_ids": TOKEN_IDS, "messages": MM_MESSAGES}, timeout=5
    )
    assert out == HS_PATH
    chat = captured[0]
    assert chat["url"].endswith("/chat/completions")
    assert chat["body"]["add_generation_prompt"] is False
    assert chat["body"]["return_token_ids"] is True

    # pre-tokenized completions path
    captured.clear()
    out = generate_hidden_states(client, "m", {"input_ids": TOKEN_IDS}, timeout=5)
    assert out == HS_PATH
    comp = captured[0]
    assert comp["url"].endswith("/completions")
    assert comp["body"]["prompt"] == TOKEN_IDS
    assert "add_generation_prompt" not in comp["body"]
