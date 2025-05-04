import json
import os
import time
import uuid
from typing import Any

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
            if "<tool_call>"  or "</tool_call>"  in new_text_chunk:
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



@app.post("/chat/completions")
async def chat_completions(request: Request):
    """以更宽松的方式处理聊天完成请求并转发给TPU端点"""
    try:
        # 直接读取原始请求数据
        body = await request.json()
    except Exception:
        # 如果JSON解析失败，则尝试读取原始body
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except:
            # 如果仍然失败，使用空字典
            body = {}

    # 宽松地获取参数，使用默认值
    model = body.get("model", "qwen2")
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", None)
    stream = body.get("stream", True)
    enable_thinking=body.get('enable_thinking',True)

    # 获取消息
    messages = body.get("messages", [])
    if not messages and "prompt" in body:
        # 如果使用了prompt字段，转换为消息格式
        messages = [{"role": "user", "content": body["prompt"]}]



    # 构建内部API请求
    chat_request = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "enable_thinking":enable_thinking
    }
    print(body.keys(),body.get('tools',True),chat_request['stream'])



    # 从队列中获取一个 TPU 端点
    endpoint = app.tpu_endpoints_queue.get()
    # 将端点放回队列以便下次使用
    app.tpu_endpoints_queue.put(endpoint)

    ip = endpoint['ipAddress']
    base_url = f'http://{ip}:8000'
    print(f"使用 TPU 端点: {base_url}")
    internal_url = f"{base_url}/api/chat"
    print(f"请求转发到: {internal_url}")

    # http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=None))

    try:
        # 如果是流式请求
        if stream:
            # 创建流式转发生成器
            async def forward_stream():
                http_client2 = httpx.AsyncClient(timeout=httpx.Timeout(timeout=None))
                # 创建唯一ID
                completion_id = f"chatcmpl-{uuid.uuid4()}"
                created = int(time.time())

                try:
                    # 使用应用级别的HTTP客户端
                    async with http_client2.stream(
                            'POST',
                            internal_url,
                            json=chat_request,
                            headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status_code != 200:
                            # 如果内部API返回错误，生成错误响应
                            error_data = {
                                "error": {
                                    "message": f"内部服务错误: {response.status_code}",
                                    "type": "internal_error",
                                    "code": response.status_code
                                }
                            }
                            yield f"data: {json.dumps(error_data)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        # 发送起始部分
                        start_data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None
                                }
                            ]
                        }
                        # yield f"data: {json.dumps(start_data)}\n\n"

                        # 处理响应流
                        accumulated_text = ""
                        async for chunk in response.aiter_text():
                            if chunk:
                                # 处理内部响应
                                print(f"收到响应块: {chunk}")

                                # 构建OpenAI格式的响应块
                                data = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": chunk},
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                                accumulated_text += chunk

                        # 发送结束标记
                        end_data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(end_data)}\n\n"

                        # 发送流结束标记
                        yield "data: [DONE]\n\n"

                except Exception as e:
                    print(f"流处理过程中出错: {str(e)}")
                    error_data = {
                        "error": {
                            "message": f"流处理错误: {str(e)}",
                            "type": "stream_error",
                            "code": 500
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"

            # 返回流式响应
            return StreamingResponse(
                forward_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
                }
            )
        else:
            http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=None))
            # 非流式请求
            async with http_client.stream(
                    'POST',
                    internal_url,
                    json=chat_request,
                    headers={"Content-Type": "application/json"}
            ) as response:
                text=''

                async for chunk in response.aiter_text():
                    if chunk:
                        text+=chunk


                # 获取响应内容并转换为OpenAI格式
                response_data = text

                # 转换为OpenAI格式的非流式响应
                completion_id = f"chatcmpl-{uuid.uuid4()}"
                openai_response = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_data#response_data.get("response", "")
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    # "usage": response_data.get("usage", {
                    #     "prompt_tokens": 0,
                    #     "completion_tokens": 0,
                    #     "total_tokens": 0
                    # })
                }

                return openai_response

    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        # 返回OpenAI格式的错误响应
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }
        )


    # finally:
    #     await http_client.aclose()
    #     print('FastAPI服务已关闭')


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