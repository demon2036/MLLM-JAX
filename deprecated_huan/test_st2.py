import streamlit as st
import jax
import jax.numpy as jnp
import numpy as np

# Assuming these functions are defined elsewhere or imported
from app.test_qwen import get_jax_mesh2, get_model, SampleState

st.set_page_config(page_title="我的应用", layout="wide")

@st.cache_resource
def load_model():
    max_cache_length = 8192
    mesh = get_jax_mesh2("1,1,-1")
    model, params, tokenizer, cache = get_model(mesh, max_cache_length=max_cache_length)

    def infer(sample_state: SampleState, params):
        i = sample_state.decoding_step
        last_token = sample_state.token_buffer[:, i].reshape((sample_state.token_buffer.shape[0], 1))

        logits, cache = model.apply({'params': params}, input_ids=last_token,
                                    position_ids=sample_state.positions,
                                    attention_mask=sample_state.attention_mask,
                                    cache=sample_state.cache)

        next_token_predict = jnp.where(i < sample_state.num_input_tokens - 1,
                                       sample_state.token_buffer[:, i + 1],
                                       jnp.argmax(logits[:, -1], axis=1)
                                       )

        sample_state.attention_mask = sample_state.attention_mask.at[:, i + 1].set(1)
        sample_state.positions += 1
        sample_state.token_buffer = sample_state.token_buffer.at[:, i + 1].set(next_token_predict)
        sample_state.next_token_buffer = next_token_predict
        sample_state.decoding_step += 1
        sample_state.cache = cache

        return sample_state




    return model, params, tokenizer, cache, max_cache_length, jax.jit(infer)


def main2():
    model, params, tokenizer, init_cache, max_cache_length, jit_infer = load_model()
    st.title("Qwen2 Chat Interface")

    # 初始化session状态
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.6
    if 'max_length' not in st.session_state:
        st.session_state.max_length = 8192

    # 生成内容容器
    output_container = st.empty()

    # 使用CSS固定输入框、按钮和参数调整在底部
    st.markdown(
        """
        <style>
        .fixed-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 20px;
            background-color: white;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }
        .generation-container {
            max-height: 60vh;
            overflow-y: auto;
            padding: 20px;
            position: fixed;
            top: 100px;
            left: 0;
            right: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 将输入框、按钮和参数调整放在固定容器中
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_area(
                "Enter your message:",
                height=150,
                key="user_input"
            )

        with col2:
            # 调整参数
            st.subheader("Parameters")
            temp_col1, temp_col2 = st.columns([3, 1])
            with temp_col1:
                st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    key="temp_slider",
                    on_change=lambda: st.session_state.update(temperature=st.session_state.temp_slider),
                )
            with temp_col2:
                st.number_input(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    key="temp_input",
                    on_change=lambda: st.session_state.update(temperature=st.session_state.temp_input),
                )

            len_col1, len_col2 = st.columns([3, 1])
            with len_col1:
                st.slider(
                    "Max Length",
                    min_value=10,
                    max_value=max_cache_length,
                    value=st.session_state.max_length,
                    step=max(1, max_cache_length // 100),
                    key="max_len_slider",
                    on_change=lambda: st.session_state.update(max_length=st.session_state.max_len_slider),
                )
            with len_col2:
                st.number_input(
                    "Max Length",
                    min_value=10,
                    max_value=max_cache_length,
                    value=st.session_state.max_length,
                    step=1,
                    key="max_len_input",
                    on_change=lambda: st.session_state.update(max_length=st.session_state.max_len_input),
                )

            if st.button("Generate"):
                if not user_input:
                    st.warning("Please enter a message.")
                    return

                # 生成内容
                with st.spinner("Generating response..."):
                    # 从session_state获取最新参数值
                    temperature = st.session_state.temperature
                    max_length = st.session_state.max_length

                    # Prepare chat template
                    messages = [{"role": "user", "content": user_input}]
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    # Tokenize input
                    inputs = tokenizer(prompt, return_tensors="jax")
                    input_ids = inputs['input_ids']
                    seq_length = input_ids.shape[1]

                    # Initialize padding and cache
                    pad_attention = jnp.zeros_like(inputs['attention_mask']).at[:, 0].set(1)
                    input_ids_pad = jnp.pad(
                        input_ids,
                        ((0, 0), (0, max_cache_length - seq_length)),
                        constant_values=tokenizer.eos_token_id
                    )
                    pad_attention = jnp.pad(
                        pad_attention,
                        ((0, 0), (0, max_cache_length - seq_length))
                    )

                    # Initialize generation state
                    position_ids = jnp.arange(0, 1)[None, ...]
                    cache = init_cache
                    key = jax.random.PRNGKey(1)
                    res = []
                    exit_token_id = tokenizer.eos_token_id

                    # 使用output_container显示生成内容
                    output_placeholder = output_container.empty()
                    progress_bar = st.progress(0)


                    sample_state = SampleState(decoding_step=0, num_input_tokens=seq_length, token_buffer=input_ids_pad,
                                               positions=position_ids, cache=cache, attention_mask=pad_attention,
                                               next_token_buffer=jnp.zeros((pad_attention.shape[0])))


                    # JIT compile once
                    for i in range(max_length):
                        sample_state = jit_infer(sample_state, params)
                        select_ids = sample_state.next_token_buffer
                        res.append(select_ids)
                        # Update progress
                        progress = (i + 1) / max_length
                        progress_bar.progress(min(progress, 1.0))

                        # Display intermediate results
                        if (i + 1) % 50 == 0 or select_ids[0] == exit_token_id:
                            current_tokens = np.array(res).reshape(-1)
                            current_text = tokenizer.decode(
                                current_tokens,
                                skip_special_tokens=False
                            )
                            # 使用Markdown显示并滚动
                            output_placeholder.markdown(
                                f"**Partial Output:**\n\n{current_text}",
                                unsafe_allow_html=True
                            )
                            # 滚动到最新内容
                            st.markdown(
                                """
                                <script>
                                document.querySelector('.generation-container').scrollTop = 
                                document.querySelector('.generation-container').scrollHeight;
                                </script>
                                """,
                                unsafe_allow_html=True
                            )

                        # Early stopping
                        if select_ids[0] == exit_token_id:
                            break

                    # Final output
                    final_tokens = np.array(res).reshape(-1)
                    final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
                    st.success("Generation complete!")
                    output_placeholder.markdown(f"**Final Output:**\n\n{final_text}")
                    # 最终滚动
                    st.markdown(
                        """
                        <script>
                        document.querySelector('.generation-container').scrollTop = 
                        document.querySelector('.generation-container').scrollHeight;
                        </script>
                        """,
                        unsafe_allow_html=True
                    )

    # 将生成内容放在可滚动的容器中
    with st.container():
        st.markdown(
            """
            <div class="generation-container">
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main2()
