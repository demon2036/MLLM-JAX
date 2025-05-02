import asyncio
import json
import os
import queue
import time
import uuid

# 假设以下函数和类在 test_qwen 中定义好
import cloud_tpu_client
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# TPU名称设置
tpu_name = 'node-1' if os.getenv('TPU_NAME') is None else os.getenv('TPU_NAME')


def tpu_endpoints_queue():
    print(f"机器名称: {tpu_name}")
    # 初始化TPU客户端
    ctc = cloud_tpu_client.client.Client(tpu_name)
    endpoints = ctc.network_endpoints()[:8]
    print(f"获取到TPU端点: {endpoints}")
    # 创建端点队列
    endpoints_queue = queue.Queue()
    # 将所有端点加入队列
    for endpoint in endpoints:
        endpoints_queue.put(endpoint)
    # 打印队列大小确认
    print(f"队列大小: {endpoints_queue.qsize()}")
    return endpoints_queue


app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 在应用启动时加载模型及相关资源
@app.on_event("startup")
async def startup_event():
    app.tpu_endpoints_queue = tpu_endpoints_queue()
    # 创建一个长期存在的HTTP客户端
    app.http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=None))
    print('FastAPI服务已启动')


# 在应用关闭时关闭HTTP客户端
@app.on_event("shutdown")
async def shutdown_event():
    await app.http_client.aclose()
    print('FastAPI服务已关闭')

# @app.post("/v1/chat/completions")
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
    extra_body=body.get('extra_body',dict())

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
        "enable_thinking":extra_body.get('enable_thinking',True)
    }

    print(bobody.keys())



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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tes_server4:app", host="0.0.0.0", port=8001, reload=True)