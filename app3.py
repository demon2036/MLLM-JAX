import json
import os
import time
import uuid
from typing import Any
import re


import httpx
import jax
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.responses import JSONResponse

from MLLM_JAX.utils import get_jax_mesh2
from safe_decode import TextStreamer
from tes_server4 import tpu_endpoints_queue
# 假设以下函数和类在 test_qwen 中定义好
from test_qwen import get_model, Sampler
from prompt import system
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
    enable_thinking:bool=True
    tools:Any=None


# 在应用启动时加载模型及相关资源（仅加载一次）
@app.on_event("startup")
async def startup_event():
    jax.distributed.initialize()
    local_devices=jax.local_devices(process_index=jax.process_index())
    print(local_devices,jax.devices())
    # 根据实际情况修改 mesh 参数
    mesh = get_jax_mesh2("1,1,-1, 1", axis_names=('dp', 'fsdp', 'tp', 'exp'))
    model, params, tokenizer = get_model(mesh, )
    print(mesh)
    app.sampler=Sampler(model, params, tokenizer,mesh=mesh)

    app.tpu_endpoints_queue = tpu_endpoints_queue()
    # 创建一个长期存在的HTTP客户端
    app.http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=None))

    print('go')



def remove_thinking_content(text):
    # Pattern to match <think> tags and their content
    # pattern = r'<think>[\s\S]*?</think>'
    # # Replace with empty string
    # cleaned_text = re.sub(pattern, '', text)

    cleaned_text = text.replace('<think>', '')
    # Replace closing tag with empty string
    cleaned_text = cleaned_text.replace('</think>', '')

    return cleaned_text


async def generate_stream_response(chat_request: ChatRequest):
    chat_request.messages.insert(0, {"role": "system", "content": system})
    for msg in chat_request.messages:
        if isinstance(msg["content"], list):
            joined = "".join(part.get("text", "") for part in msg["content"])
            msg["content"] = joined
        msg['content']=remove_thinking_content(msg['content'])
        # if msg['role']=='developer':
        #     msg['role']='system'
        #     msg['content']=f'system:'+msg['content']




    sampler=app.sampler
    prompt = sampler.tokenizer.apply_chat_template(
        chat_request.messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=chat_request.enable_thinking,
        tools=chat_request.tools
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
            if "<tool_call>" in new_text_chunk  or "</tool_call>"  in new_text_chunk:
                print(new_text_chunk,token)

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


# @app.post("/chat/completions")
# async def chat_completions(chat_request: ChatRequest):
#     """
#     FastAPI 端点，作为中转代理，将请求转发给 /api/chat 并返回响应。
#     """
#
#     endpoint=app.tpu_endpoints_queue.get()
#
#     app.tpu_endpoints_queue.put(endpoint)
#     ip=endpoint['ipAddress']
#     base_url=f'http://{  ip}:8000'
#     print(base_url)
#
#     try:
#         # 创建一个内部请求，转发到 /api/chat
#         async with httpx.AsyncClient() as client:
#             # 假设服务器在同一主机上运行
#             # 使用相对URL或根据实际情况修改URL
#             internal_url = f"{base_url}/api/chat"
#             print(internal_url)
#
#             # 转发请求体并获取响应
#             response = await client.post(
#                 internal_url,
#                 json=chat_request.dict(),
#                 headers={"Content-Type": "application/json"}
#             )
#             # 直接将流式响应返回给客户端
#             return StreamingResponse(
#                 response.aiter_bytes(),
#                 media_type="text/event-stream",
#                 headers={"Cache-Control": "no-cache"}
#             )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




# 为错误处理添加异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"全局错误处理: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": 500
            }
        }
    )



# 添加 uvicorn 启动代码
if __name__ == "__main__":
    import uvicorn

    # 配置 uvicorn 启动参数
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))

    # 启动服务器
    uvicorn.run(
        "app3:app",  # 使用当前文件名:app实例
        host=host,
        port=port,
        reload=False,  # 生产环境建议设为False
        workers=1,  # 对于JAX模型，通常使用单个worker
        log_level="info"
    )