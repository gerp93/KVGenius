"""
Thin HTTP client for a locally-running Ollama server.
KVGenius does not load chat models in-process - it calls Ollama's REST API.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from .config import get_setting

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server can't be reached."""
    pass


def get_host() -> str:
    """Get the configured Ollama base URL."""
    return (get_setting("ollama_host", DEFAULT_HOST) or DEFAULT_HOST).rstrip("/")


def _request(method: str, path: str, timeout: float = 30, **kwargs) -> requests.Response:
    url = f"{get_host()}{path}"
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        raise OllamaUnavailableError(
            f"Ollama not reachable at {get_host()}: {e}"
        ) from e


def list_models() -> List[str]:
    """Return the tags of all models Ollama currently has pulled locally."""
    resp = _request("GET", "/api/tags")
    data = resp.json()
    return [m.get("name", "") for m in data.get("models", [])]


def is_model_available(model_tag: str) -> bool:
    """Check whether a model tag is already pulled in Ollama."""
    return model_tag in list_models()


def chat(
    model: str,
    messages: List[Dict[str, str]],
    options: Optional[Dict[str, Any]] = None,
    keep_alive: Optional[str] = None,
    timeout: float = 300,
) -> Dict[str, Any]:
    """
    Send a chat request to Ollama and return the parsed response.

    Args:
        model: Ollama model tag
        messages: List of {"role": "user"|"assistant"|"system", "content": str}
        options: Ollama generation options (temperature, top_p, top_k, repeat_penalty, num_predict, stop, ...)
        keep_alive: How long to keep the model loaded after this request (e.g. "5m", "0")
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response from Ollama (message.content is the reply text).
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    resp = _request("POST", "/api/chat", timeout=timeout, json=payload)
    return resp.json()


def unload_model(model_tag: str) -> None:
    """Ask Ollama to unload a model from VRAM immediately."""
    try:
        _request(
            "POST", "/api/generate", timeout=30,
            json={"model": model_tag, "prompt": "", "keep_alive": 0},
        )
    except OllamaUnavailableError:
        # Nothing to unload if the server isn't reachable
        pass


def pull(
    model_tag: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[bool, str]:
    """
    Pull (download) a model into Ollama, streaming progress.

    Returns:
        Tuple of (success: bool, message: str)
    """
    url = f"{get_host()}/api/pull"
    try:
        with requests.post(url, json={"model": model_tag, "stream": True}, stream=True, timeout=3600) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if evt.get("error"):
                    return False, evt["error"]

                status = evt.get("status", "")
                total = evt.get("total")
                completed = evt.get("completed")
                if progress_callback:
                    if total and completed is not None:
                        progress_callback(min(completed / total, 1.0), status)
                    else:
                        progress_callback(0.0, status)

        return True, f"Successfully pulled {model_tag}"
    except requests.exceptions.RequestException as e:
        raise OllamaUnavailableError(f"Ollama not reachable at {get_host()}: {e}") from e
