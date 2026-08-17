"""
Thin HTTP client for a locally-running ComfyUI server.
KVGenius does not load diffusion pipelines in-process - it submits workflow
graphs to ComfyUI's queue API and polls for the result.
"""
import copy
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from .config import CONFIG_DIR, get_setting

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:8188"
WORKFLOWS_DIR = CONFIG_DIR / "comfyui_workflows"

# Node IDs used by the shipped workflow templates in config/comfyui_workflows/.
# If you edit those JSON files, keep these in sync.
NODE_MAP = {
    "checkpoint": "4",
    "positive": "6",
    "negative": "7",
    "sampler": "3",
    "latent": "5",
    "lora": "10",  # inserted dynamically, not present in the base templates
}


class ComfyUIUnavailableError(RuntimeError):
    """Raised when the ComfyUI server can't be reached."""
    pass


class GenerationCancelled(Exception):
    """Raised internally when a caller-supplied cancel_callback returns True."""
    pass


def get_host() -> str:
    """Get the configured ComfyUI base URL."""
    return (get_setting("comfyui_host", DEFAULT_HOST) or DEFAULT_HOST).rstrip("/")


def _request(method: str, path: str, timeout: float = 30, **kwargs) -> requests.Response:
    url = f"{get_host()}{path}"
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        raise ComfyUIUnavailableError(
            f"ComfyUI not reachable at {get_host()}: {e}"
        ) from e


def is_available() -> bool:
    """Check whether the ComfyUI server is reachable."""
    try:
        _request("GET", "/system_stats", timeout=5)
        return True
    except ComfyUIUnavailableError:
        return False


def get_object_info(node_class: Optional[str] = None) -> Dict[str, Any]:
    """Query ComfyUI's /object_info for available node types and their inputs."""
    path = f"/object_info/{node_class}" if node_class else "/object_info"
    resp = _request("GET", path, timeout=15)
    return resp.json()


def list_checkpoints() -> List[str]:
    """List checkpoint filenames ComfyUI's CheckpointLoaderSimple currently sees."""
    info = get_object_info("CheckpointLoaderSimple")
    try:
        return info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    except (KeyError, IndexError, TypeError):
        return []


def load_workflow(family: str) -> Dict[str, Any]:
    """Load a workflow template JSON by pipeline family (e.g. 'sd15', 'sdxl', 'flux')."""
    path = WORKFLOWS_DIR / f"{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"No ComfyUI workflow template for family '{family}': {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _substitute_strings(node: Dict[str, Any], replacements: Dict[str, str]) -> None:
    for key, value in list(node.get("inputs", {}).items()):
        if isinstance(value, str) and value in replacements:
            node["inputs"][key] = replacements[value]


def patch_workflow(
    template: Dict[str, Any],
    ckpt_name: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
) -> Dict[str, Any]:
    """Return a deep copy of a workflow template with the given parameters filled in."""
    workflow = copy.deepcopy(template)

    replacements = {
        "__CKPT_NAME__": ckpt_name,
        "__PROMPT__": prompt,
        "__NEGATIVE_PROMPT__": negative_prompt,
    }
    for node in workflow.values():
        _substitute_strings(node, replacements)

    sampler = workflow[NODE_MAP["sampler"]]["inputs"]
    sampler["steps"] = steps
    sampler["cfg"] = cfg
    sampler["seed"] = seed
    if sampler_name:
        sampler["sampler_name"] = sampler_name
    if scheduler:
        sampler["scheduler"] = scheduler

    latent = workflow[NODE_MAP["latent"]]["inputs"]
    latent["width"] = width
    latent["height"] = height

    if lora_name:
        ckpt_id = NODE_MAP["checkpoint"]
        lora_id = NODE_MAP["lora"]
        workflow[lora_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
                "model": [ckpt_id, 0],
                "clip": [ckpt_id, 1],
            },
        }
        sampler["model"] = [lora_id, 0]
        workflow[NODE_MAP["positive"]]["inputs"]["clip"] = [lora_id, 1]
        workflow[NODE_MAP["negative"]]["inputs"]["clip"] = [lora_id, 1]

    return workflow


def submit(workflow: Dict[str, Any], client_id: Optional[str] = None) -> str:
    """Queue a workflow for execution. Returns the prompt_id."""
    client_id = client_id or str(uuid.uuid4())
    resp = _request("POST", "/prompt", json={"prompt": workflow, "client_id": client_id})
    data = resp.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise ComfyUIUnavailableError(f"ComfyUI did not return a prompt_id: {data}")
    return prompt_id


def get_history(prompt_id: str) -> Dict[str, Any]:
    """Get the history entry for a prompt_id, or {} if not finished yet."""
    resp = _request("GET", f"/history/{prompt_id}")
    return resp.json().get(prompt_id, {})


def cancel(prompt_id: str) -> None:
    """Best-effort cancel: interrupt if currently running, and remove from queue if still pending."""
    try:
        _request("POST", "/interrupt", timeout=10)
    except ComfyUIUnavailableError:
        pass
    try:
        _request("POST", "/queue", timeout=10, json={"delete": [prompt_id]})
    except ComfyUIUnavailableError:
        pass


def fetch_image(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    """Download generated image bytes from ComfyUI's /view endpoint."""
    resp = _request(
        "GET", "/view", timeout=60,
        params={"filename": filename, "subfolder": subfolder, "type": folder_type},
    )
    return resp.content


def wait_for_result(
    prompt_id: str,
    poll_interval: float = 1.0,
    timeout: float = 600,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Poll /history until the prompt finishes. Returns the history entry.
    Raises GenerationCancelled if cancel_callback() returns True while waiting,
    or TimeoutError if the prompt doesn't finish within `timeout` seconds.
    """
    start = time.time()
    while True:
        if cancel_callback and cancel_callback():
            cancel(prompt_id)
            raise GenerationCancelled("Generation cancelled by user")

        entry = get_history(prompt_id)
        if entry:
            return entry

        if time.time() - start > timeout:
            raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish within {timeout}s")

        time.sleep(poll_interval)


def generate(
    family: str,
    ckpt_name: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    timeout: float = 600,
) -> bytes:
    """
    Submit a txt2img generation to ComfyUI and return the resulting PNG bytes.

    progress_callback reports coarse (queued/running vs done) progress, not
    per-step - true per-step progress would need ComfyUI's websocket feed,
    which this client intentionally doesn't use.
    """
    template = load_workflow(family)
    workflow = patch_workflow(
        template, ckpt_name=ckpt_name, prompt=prompt, negative_prompt=negative_prompt,
        steps=steps, cfg=cfg, width=width, height=height, seed=seed,
        sampler_name=sampler_name, scheduler=scheduler,
        lora_name=lora_name, lora_strength=lora_strength,
    )

    prompt_id = submit(workflow)
    if progress_callback:
        progress_callback(0, steps)

    entry = wait_for_result(prompt_id, cancel_callback=cancel_callback, timeout=timeout)

    if progress_callback:
        progress_callback(steps, steps)

    outputs = entry.get("outputs", {})
    for node_output in outputs.values():
        images = node_output.get("images")
        if images:
            img = images[0]
            return fetch_image(img["filename"], img.get("subfolder", ""), img.get("type", "output"))

    raise RuntimeError(f"ComfyUI prompt {prompt_id} finished with no image output: {entry}")
