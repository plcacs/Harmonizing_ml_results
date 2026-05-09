import gc
import os
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-31B-it")
TORCH_DTYPE = torch.bfloat16
DEVICE_MAP = os.environ.get("GEMMA4_31B_DEVICE_MAP", "auto")
MAX_INPUT_TOKENS_ENV = os.environ.get("GEMMA4_31B_MAX_INPUT_TOKENS")
MAX_INPUT_TOKENS = int(MAX_INPUT_TOKENS_ENV) if MAX_INPUT_TOKENS_ENV else None
GENERATE_LOCK = threading.Lock()


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


app = FastAPI(title="Gemma 4 Transformers Server")

print(f"Loading tokenizer: {MODEL_ID}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading model: {MODEL_ID}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
    device_map=DEVICE_MAP,
)
model.eval()
print(f"Model loaded with device_map={DEVICE_MAP}", flush=True)


def normalize_content(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif isinstance(item, dict):
            parts.append(str(item))
        else:
            parts.append(str(item))
    return "\n".join(part for part in parts if part)


def input_device() -> torch.device:
    return next(model.parameters()).device


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device_map": DEVICE_MAP,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "max_input_tokens": MAX_INPUT_TOKENS,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "huggingface",
            }
        ],
    }


@app.post("/v1/chat/completions")
@torch.inference_mode()
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.model and request.model != MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Requested model {request.model!r} does not match loaded model "
                f"{MODEL_ID!r}."
            ),
        )

    messages = [
        {"role": msg.role, "content": normalize_content(msg.content)}
        for msg in request.messages
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    with GENERATE_LOCK:
        inputs: dict[str, torch.Tensor] | None = None
        outputs: torch.Tensor | None = None
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=MAX_INPUT_TOKENS is not None,
                max_length=MAX_INPUT_TOKENS,
            )
            inputs = {key: value.to(input_device()) for key, value in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            created = int(time.time())

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": int(inputs["input_ids"].shape[1]),
                    "completion_tokens": int(generated_ids.shape[0]),
                    "total_tokens": int(outputs.shape[1]),
                },
            }
        except torch.OutOfMemoryError as exc:
            raise HTTPException(status_code=507, detail=f"CUDA out of memory: {exc}") from exc
        finally:
            del outputs
            del inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
