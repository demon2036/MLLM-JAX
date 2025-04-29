import json
from datetime import datetime

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

tpu_name='node-2'

# ctc=cloud_tpu_client.client.Client(tpu_name)
# endpoints=ctc.network_endpoints()
# print(endpoints)






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


    max_cache_length = 1024
    # 根据实际情况修改 mesh 参数
    mesh = get_jax_mesh2("1,1,-1")
    model, params, tokenizer, init_cache = get_model(mesh, max_cache_length=max_cache_length)
    del init_cache
    print(mesh)
    app.sampler=Sampler(model, params, tokenizer,mesh=mesh)
    print('go')


    # max_cache_length = 4096
    # # # 根据实际情况修改 mesh 参数
    # mesh = get_jax_mesh2("1,1,-1")
    # model_path = 'Qwen/Qwen2.5-7B-Instruct'
    # model, params, tokenizer = get_model(mesh, model_path=model_path, )
    # sampler = Sampler(model, tokenizer,mesh=mesh,)
    # app.sampler=sampler
    # app.params=params


# 检查是否生成结束 token
# data = {
#     "model": "qwen2.5:7b-instruct",
#     "created_at": datetime.utcnow().isoformat() + "Z",
#     "response": f" {current_text}",
#     "done": "False"
# }


async def generate_stream_response(chat_request: ChatRequest):
    # print(chat_request)
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

    res_tokens=[]

    generated_token_ids=[]
    all_decoded_text: str = ""  #
    decoded_text_offset=0
    decoder=TextStreamer(tokenizer=sampler.tokenizer)


    def handle_msg(chunk):
        # return {
        #       "model": "qwen2.5:7b-instruct",
        #       "created_at":datetime.utcnow().isoformat() + "Z",
        #       "message": {
        #         "role": "assistant",
        #         "content": f"{chunk}",
        #         "images": None,
        #         "tool_calls": []
        #       },
        #       "done": False,
        #     }

        return      {
          "id": "chatcmpl-935",
          "object": "chat.completion.chunk",
          "created": 1745666537,
          "model": "qwen2.5:7b-instruct",
          "system_fingerprint": "fp_ollama",
          "choices": [
            {
              "index": 0,
              "delta": {
                "role": "assistant",
                "content": chunk
              },
              "finish_reason": None
            }
          ]
        }



    # if "r1" in chat_request.model:
    #     data= handle_msg('<think>')
    #     yield f"{json.dumps(data, ensure_ascii=False)}\n"
    # else:
    #     print('this is not r1 model')

    l = sampler.tokenizer(prompt, return_tensors="jax", )['input_ids'].shape[1]
    max_length=min(sampler.find_ceil(l)*2,int(16384*1.5))


    async for token in sampler.generate_prefill_auto_regressive(prompt,max_length=max_length,stream=True):
        generated_token_ids.append( int(token[0]))
        token=int(token[0])
        new_text_chunk = decoder.put(np.array([token]))
        if new_text_chunk is not None:
            print(new_text_chunk)
            data=handle_msg(new_text_chunk)
            # yield f"{json.dumps(data, ensure_ascii=False)}\n"
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    if final_chunk := decoder.end():
        print(final_chunk)
        data = handle_msg(final_chunk)
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        # data = handle_msg(final_chunk)
        # yield f"{json.dumps(data, ensure_ascii=False)}\n"
        # data={
        #     "model": "qwen2.5:7b-instruct",
        #     "created_at": datetime.utcnow().isoformat() + "Z",
        #     "message": {
        #         "role": "assistant",
        #         "content": f"{final_chunk}",
        #         "images": None,
        #         "tool_calls": []
        #     },
        #     "done": True,
        # }
        # yield f"{json.dumps(data, ensure_ascii=False)}\n"
    yield 'data: [DONE]\n\n'






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
