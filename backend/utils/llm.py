import httpx
import json
from backend.utils.config import OLLAMA_BASE_URL, OLLAMA_MODEL


async def generate_response(system_prompt: str, user_prompt: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": True,
            },
            timeout=120.0,
        )
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue


def generate_response_sync(system_prompt: str, user_prompt: str) -> str:
    """Non-streaming version for testing."""

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]