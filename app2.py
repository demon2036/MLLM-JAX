import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel

from MLLM_JAX.language.qwen2.configuration_qwen2 import init_cache
# 假设以下函数和类在 test_qwen 中定义好
from test_qwen import get_jax_mesh2, get_model, SampleState, Sampler, create_sample_state
import cloud_tpu_client

tpu_name='node-2'

ctc=cloud_tpu_client.client.Client(tpu_name)
endpoints=ctc.network_endpoints()
print(endpoints)






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
    max_tokens: int = 16384
    stream: bool = True


# 在应用启动时加载模型及相关资源（仅加载一次）
@app.on_event("startup")
async def startup_event():
    pass
    # max_cache_length = 16384
    # # 根据实际情况修改 mesh 参数
    # mesh = get_jax_mesh2("1,1,-1")
    # model, params, tokenizer, init_cache = get_model(mesh, max_cache_length=max_cache_length)
    # del init_cache
    # print(mesh)
    # app.sampler=Sampler(model, params, tokenizer,mesh=mesh)
    # print('go')


async def generate_stream_response(chat_request: ChatRequest):
    sampler=app.sampler
    prompt = sampler.tokenizer.apply_chat_template(
        chat_request.messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(chat_request.max_tokens)

    res_tokens=[]
    for i,token in enumerate(sampler.generate_prefill_auto_regressive(prompt,max_length=chat_request.max_tokens,stream=True)):
        res_tokens.append(token)
        if (i + 1) % 100 == 0:
            current_text = sampler.tokenizer.decode(
                # np.array(next_token),
                np.array(res_tokens).reshape(-1),
                skip_special_tokens=True
            )
            # res_tokens = []
            # 检查是否生成结束 token
            # SSE 格式（每个消息前缀 "data:"），确保客户端能实时接收
            # yield f"{current_text}"
            # await asyncio.sleep(0.0001)  # 根据需要调节间隔

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


    print()



    while True:
        pass



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
    print(chat_request)
    try:
        return StreamingResponse(
            generate_stream_response(chat_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
