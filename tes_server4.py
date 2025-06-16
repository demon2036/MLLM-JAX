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

from prompt import system_prompt

# TPU名称设置
tpu_name = 'node-2' if os.getenv('TPU_NAME') is None else os.getenv('TPU_NAME')
print(tpu_name)


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
def create_response_data(completion_id, created, model, content=None, function_call=None, finish_reason="stop",
                         is_delta=False):
    """
    创建统一格式的响应数据

    Args:
        completion_id: 完成ID
        created: 创建时间戳
        model: 模型名称
        content: 响应内容，可以为None
        function_call: 函数调用数据，可以为None
        finish_reason: 完成原因
        is_delta: 是否为流式响应的delta格式

    Returns:
        dict: 格式化的响应数据
    """
    if is_delta:
        # 流式响应格式
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None  # 开始时设为None，根据情况更新
                }
            ]
        }

        # 添加内容到delta
        if content is not None:
            data["choices"][0]["delta"]["content"] = content

        # 添加工具调用到delta
        if function_call is not None:
            # 确保tool_calls是列表
            if not isinstance(function_call, list):
                function_call = [function_call]

            # 工具调用使用tool_calls字段
            data["choices"][0]["delta"]["tool_calls"] = function_call

            # 如果是最后一个工具调用块，则设置finish_reason
            if finish_reason == "function_call":
                data["choices"][0]["finish_reason"] = "tool_call"

        # 如果不是工具调用且明确指定finish_reason，则设置它
        elif finish_reason != "stop":
            data["choices"][0]["finish_reason"] = finish_reason
    else:
        # 非流式响应格式
        data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content if content is not None else None
                    },
                    "finish_reason": finish_reason
                }
            ]
        }

        # 如果有工具调用，添加到消息中
        if function_call is not None:
            # 确保tool_calls是列表
            if not isinstance(function_call, list):
                function_call = [function_call]

            # 非流式响应中使用tool_calls字段
            data["choices"][0]["message"]["tool_calls"] = function_call
            # 设置finish_reason
            if finish_reason == "function_call":
                data["choices"][0]["finish_reason"] = "tool_call"

    return data






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
    tools=body.get('tools',None)

    # 获取消息
    messages = body.get("messages", [])
    if not messages and "prompt" in body:
        # 如果使用了prompt字段，转换为消息格式
        messages = [{"role": "user", "content": body["prompt"]}]



    # messages.insert(0, {"role": "system", "content": system_prompt})
    for msg in messages:
        if msg['role']=='tool':
            content=json.loads(msg['content'])
            for result in content:
                if 'thumbnail' in result:
                    del result['thumbnail']
            msg['content']=json.dumps(content,ensure_ascii=False)

    # print(messages)

    # 构建内部API请求
    chat_request = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "enable_thinking":enable_thinking,
        "tools":tools
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
                completion_id = f"chatcmpl-{uuid.uuid4()}"
                created = int(time.time())

                try:
                    async with http_client2.stream(
                            'POST',
                            internal_url,
                            json=chat_request,
                            headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status_code != 200:
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
                        start_data = create_response_data(
                            completion_id=completion_id,
                            created=created,
                            model=model,
                            is_delta=True
                        )
                        start_data["choices"][0]["delta"]["role"] = "assistant"
                        yield f"data: {json.dumps(start_data)}\n\n"

                        # 处理响应流
                        accumulated_text = ""
                        in_tool_call = False
                        tool_call_content = ""

                        async for chunk in response.aiter_text():
                            if chunk:
                                print(f"收到响应块: {chunk}")

                                # 检查是否是工具调用的开始或结束
                                if "<tool_call>" in chunk:
                                    in_tool_call = True
                                    # 分割处理前面的正常内容
                                    normal_content = chunk.split("<tool_call>")[0]
                                    if normal_content:
                                        data = create_response_data(
                                            completion_id=completion_id,
                                            created=created,
                                            model=model,
                                            content=normal_content,
                                            is_delta=True
                                        )
                                        yield f"data: {json.dumps(data)}\n\n"
                                        accumulated_text += normal_content

                                    # 开始收集工具调用内容
                                    tool_call_content = chunk.split("<tool_call>")[1]
                                    continue

                                if in_tool_call and "</tool_call>" in chunk:
                                    in_tool_call = False
                                    # 添加结束前的内容
                                    tool_call_content += chunk.split("</tool_call>")[0]

                                    try:
                                        # 直接解析JSON对象
                                        tool_call_data = json.loads(tool_call_content.strip())

                                        if not isinstance(tool_call_data, list):
                                            function_args = tool_call_data.get("arguments", "")
                                            if not isinstance(function_args, str):
                                                function_args = json.dumps(function_args,ensure_ascii=False)

                                            # 如果不是列表，将其转换为列表中的一项
                                            tool_call_list = [{
                                                "id": tool_call_data.get("id", f"call_{uuid.uuid4()}"),
                                                "type": tool_call_data.get("type", "function"),
                                                "index": 0,
                                                "function": {
                                                    "name": tool_call_data.get("name", ""),
                                                    "arguments": function_args
                                                }
                                            }]


                                        # 发送函数调用格式的 delta
                                        tool_call_data = create_response_data(
                                            completion_id=completion_id,
                                            created=created,
                                            model=model,
                                            function_call=tool_call_list,
                                            finish_reason="function_call",
                                            is_delta=True
                                        )
                                        print(tool_call_data)
                                        yield f"data: {json.dumps(tool_call_data,ensure_ascii=False)}\n\n"

                                        # 继续处理结束标签后的内容
                                        if "</tool_call>" in chunk:
                                            remaining_content = chunk.split("</tool_call>")[1]
                                            if remaining_content:
                                                data = create_response_data(
                                                    completion_id=completion_id,
                                                    created=created,
                                                    model=model,
                                                    content=remaining_content,
                                                    is_delta=True
                                                )
                                                yield f"data: {json.dumps(data,ensure_ascii=False)}\n\n"
                                                accumulated_text += remaining_content
                                    except Exception as e:
                                        print(f"解析工具调用时出错: {str(e)}")
                                    continue

                                if in_tool_call:
                                    # 收集工具调用内容
                                    tool_call_content += chunk
                                else:
                                    # 正常内容处理
                                    data = create_response_data(
                                        completion_id=completion_id,
                                        created=created,
                                        model=model,
                                        content=chunk,
                                        is_delta=True
                                    )
                                    yield f"data: {json.dumps(data,ensure_ascii=False)}\n\n"
                                    accumulated_text += chunk

                        # 发送结束标记
                        finish_reason = "function_call" if in_tool_call else "stop"
                        end_data = create_response_data(
                            completion_id=completion_id,
                            created=created,
                            model=model,
                            finish_reason=finish_reason,
                            is_delta=True
                        )
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
            # The modified non-stream handling section in the chat_completions function:

        else:
            http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=None))
            # 非流式请求
            async with http_client.stream(
                    'POST',
                    internal_url,
                    json=chat_request,
                    headers={"Content-Type": "application/json"}
            ) as response:
                text = ''
                tool_call_data = None
                in_tool_call = False
                tool_call_content = ""

                async for chunk in response.aiter_text():
                    if chunk:
                        # 检查是否是工具调用的开始或结束
                        if "<tool_call>" in chunk:
                            in_tool_call = True
                            # 分割处理前面的正常内容
                            normal_content = chunk.split("<tool_call>")[0]
                            if normal_content:
                                text += normal_content
                            # 开始收集工具调用内容
                            tool_call_content = chunk.split("<tool_call>")[1]
                            continue

                        if in_tool_call and "</tool_call>" in chunk:
                            in_tool_call = False
                            # 添加结束前的内容
                            tool_call_content += chunk.split("</tool_call>")[0]

                            try:
                                # 直接解析JSON对象
                                tool_call_data = json.loads(tool_call_content.strip())

                                if not isinstance(tool_call_data, list):
                                    function_args = tool_call_data.get("arguments", "")
                                    if not isinstance(function_args, str):
                                        function_args = json.dumps(function_args, ensure_ascii=False)

                                    # 如果不是列表，将其转换为列表中的一项
                                    tool_call_data = [{
                                        "id": tool_call_data.get("id", f"call_{uuid.uuid4()}"),
                                        "type": tool_call_data.get("type", "function"),
                                        "index": 0,
                                        "function": {
                                            "name": tool_call_data.get("name", ""),
                                            "arguments": function_args
                                        }
                                    }]

                            except Exception as e:
                                print(f"解析工具调用时出错: {str(e)}")

                            # 继续处理结束标签后的内容
                            if "</tool_call>" in chunk:
                                remaining_content = chunk.split("</tool_call>")[1]
                                if remaining_content:
                                    text += remaining_content
                            continue

                        if in_tool_call:
                            # 收集工具调用内容
                            tool_call_content += chunk
                        else:
                            # 正常内容处理
                            text += chunk

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
                                "content": text if not tool_call_data else None
                            },
                            "finish_reason": "tool_call" if tool_call_data else "stop"
                        }
                    ]
                }

                # 如果有工具调用，添加到消息中
                if tool_call_data:
                    openai_response["choices"][0]["message"]["tool_calls"] = tool_call_data

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