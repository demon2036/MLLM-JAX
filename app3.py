import json
import os
import queue
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel
from transformers import TextIteratorStreamer, AutoTokenizer

from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache
# 假设以下函数和类在 test_qwen 中定义好
from test_qwen import get_model, Sampler
import cloud_tpu_client

from MLLM_JAX.utils import get_jax_mesh2
from safe_decode import  TextStreamer

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求体定义
class ChatRequest(BaseModel):
    model: str = "qwen2"  # 示例模型名称
    messages: list  # 形如 [{"role": "user", "content": "xxx"}, ...]
    temperature: float = 0.7
    max_tokens: int|None = None
    stream: bool = True


# 在应用启动时加载模型及相关资源（仅加载一次）
@app.on_event("startup")
async def startup_event():
    jax.distributed.initialize()
    local_devices=jax.local_devices(process_index=jax.process_index())
    print(local_devices,jax.devices())
    max_cache_length = 1024
    # 根据实际情况修改 mesh 参数
    mesh = get_jax_mesh2("1,1,-1",devices=local_devices)
    model, params, tokenizer, init_cache = get_model(mesh, max_cache_length=max_cache_length)
    del init_cache
    print(mesh)
    app.sampler=Sampler(model, params, tokenizer,mesh=mesh)
    print('go')


async def generate_stream_response(chat_request: ChatRequest):
    for msg in chat_request.messages:
        if isinstance(msg["content"], list):
            joined = "".join(part.get("text", "") for part in msg["content"])
            msg["content"] = joined


    sampler=app.sampler
    prompt = sampler.tokenizer.apply_chat_template(
        chat_request.messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    print(prompt)
    generated_token_ids=[]
    decoder=TextStreamer(tokenizer=sampler.tokenizer)
    l = sampler.tokenizer(prompt, return_tensors="jax", )['input_ids'].shape[1]
    max_length=min(sampler.find_ceil(l)*2,int(16384*1.5))

    async for token in sampler.generate_prefill_auto_regressive(prompt,max_length=max_length,stream=True):
        generated_token_ids.append( int(token[0]))
        token=int(token[0])
        new_text_chunk = decoder.put(np.array([token]))
        if new_text_chunk is not None:
            yield new_text_chunk
    if final_chunk := decoder.end():
        yield final_chunk


@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    """
    FastAPI 端点，根据请求参数调用模型生成，并返回流式响应。
    """
    try:
        return StreamingResponse(
            generate_stream_response(chat_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/completions")
async def chat_completions(chat_request: ChatRequest):
    """
    FastAPI 端点，作为中转代理，将请求转发给 /api/chat 并返回响应。
    """

    endpoint=app.tpu_endpoints_queue.get()

    app.tpu_endpoints_queue.put(endpoint)
    ip=endpoint['ipAddress']
    base_url=f'http://{  ip}:8000'
    print(base_url)

    try:
        # 创建一个内部请求，转发到 /api/chat
        async with httpx.AsyncClient() as client:
            # 假设服务器在同一主机上运行
            # 使用相对URL或根据实际情况修改URL
            internal_url = f"{base_url}/api/chat"
            print(internal_url)

            # 转发请求体并获取响应
            response = await client.post(
                internal_url,
                json=chat_request.dict(),
                headers={"Content-Type": "application/json"}
            )
            # 直接将流式响应返回给客户端
            return StreamingResponse(
                response.aiter_bytes(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/v1/chat/completions")
async def chat(chat_request: ChatRequest):
    """
    FastAPI 端点，根据请求参数调用模型生成，并返回流式响应。
    """
    # print(chat_request)
    try:
        return StreamingResponse(
            generate_stream_response(chat_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
