import json

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoProcessor

# 假设以下函数和类在 test_qwen 中定义好
from MLLM_JAX.multinomial_sample import get_jax_mesh2, get_model, Sampler

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

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
    max_tokens: int = 8192
    stream: bool = True


# 在应用启动时加载模型及相关资源（仅加载一次）
@app.on_event("startup")
async def startup_event():
    max_cache_length = 16384
    # 根据实际情况修改 mesh 参数
    mesh = get_jax_mesh2("1,1,-1")
    model, params, processor = get_model(mesh, max_cache_length=max_cache_length)
    print(mesh)
    app.sampler=Sampler(model, params, processor,mesh=mesh)
    print('go')


async def generate_stream_response(chat_request: ChatRequest):
    sampler=app.sampler


    messages=chat_request.messages


    for i in range(len(messages)):
        message=messages[i]
        message['content'] = [{'type': 'text', 'text': message} if isinstance(message, str) else message for message in
                               message['content']]

    model_id = "google/gemma-3-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        # return_dict=True, return_tensors="pt"
    )
    # print(inputs)



    res_tokens=[]
    for i,token in enumerate(sampler.generate_prefill_auto_regressive(messages,max_length=chat_request.max_tokens,stream=True)):
        res_tokens.append(token)
        if (i + 1) % 100 == 0:
            current_text = sampler.tokenizer.decode(
                # np.array(next_token),
                np.array(res_tokens).reshape(-1),
                skip_special_tokens=True
            )
            print(current_text)

    current_text = sampler.tokenizer.decode(
        # np.array(next_token),
        np.array(res_tokens).reshape(-1),
        skip_special_tokens=True
    )

    output={
        "choices":[{
            'finish_reason': 'stop', 'index': 0, 'logprobs': None,
            "message": {"content": current_text,'role': 'assistant', 'annotations': None, 'audio': None, 'function_call': None,
            'tool_calls': [], 'reasoning_content': None},
            'stop_reason': None
        }]
    }

    yield json.dumps(output)




@app.post("/chat")
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
