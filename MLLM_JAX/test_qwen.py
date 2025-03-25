import flax.linen
import torch
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from language.qwen2.configuration_qwen2 import Qwen2Config, init_cache
from language.llama import convert_torch_to_flax_llama, test_vicuna, test_vicuna_torch
from language.qwen2.modular_qwen2 import Qwen2ForCausalLM



content="医生，我胸椎，腰椎附近，总是觉得很累，什么情况?就是我穿衣服像是羽绒服这种比较重的就会加剧，轻的就不明显。有腰间盘突出，但是是4/5的，这应该与胸椎附近无关是吧？"
content="""翻译成中文,确保详细准确，一个字都不能遗漏。

Dear Area Chair,

We are writing to express our serious concerns **regarding the quality of ZbWb (Reviewer 2)**'s review for our submission. Based on several issues we identified: **ZbWb suggestions include misguided comparisons, factual errors, misplaced focus on our method, etc.**

The series of fundamental errors displayed by ZbWb contradicts the level of understanding expected from someone with relevant research expertise. We are highly suspicious that the review may have been generated, or at least heavily assisted, by a Large Language Model (LLM). 

-----
### **1. Strong Suspicions of LLM Usage and Factual Errors**

Our most significant concern lies in Reviewer ZbWb’s suggestion which contains glaring factual errors, further indicate a lack of expertise or possible reliance on an LLM. We strongly suspect these errors  may have originated from LLM-generated content, as LLMs are prone to generating “hallucinated” (factually incorrect) outputs that appear superficially valid but fail under scrutiny (e.g.  Weakness 5 from Reviewer ZbWb):

*"The authors should also show the results for the epsilon value of 8/255 for l_inf, l2, and l_1 attacks."*

- **Basic misunderstanding of adversarial attacks:** It is incorrect to suggest that the same epsilon value (e.g., $ 8/255 $) can be applied across $ l_{\infty} $, $ l_2 $ and $ l_1 $ norms. The strength of an attack depends on the norm, and using the same epsilon for different norms leads to vastly different attack strengths. This is fundamental knowledge in adversarial robustness research [1-3].
- **Incorrect benchmark:** For ImageNet, standard evaluations use $ l_{\infty} $ attacks with $ \epsilon = 4/255 $, not $ 8/255 $ [1-3].

The above mistakes suggest a lack of familiarity with adversarial robustness benchmarks, **which is highly unusual for a CVPR reviewer.**



------


### **2. Factual Errors Regarding Scaling and Data Usage**

ZbWb even misunderstood the core of our method. ZbWb claimed that the performance improvements in our work stem from "data scaling," as shown in one of their cited papers [4], **and thus concluded that our paper lacks contribution**. This is factually incorrect.

Our core approach relies on mitigating overfitting using our Two-Stage operations. We scaled the dataset from 1M to 100M without altering the model size or training iterations. The key lies in addressing overfitting, as simply increasing the dataset size alone would not have been sufficient to achieve our results. Unlike their cited paper, which scales datasets by increasing data iterations from 0.7B to 5B—resulting in a 7x increase in total cost—our method without incurring any additional cost.

ZbWb also misunderstood this saturation effect, incorrectly claiming that our method cannot scale further. Like we mentioned above, once overfitting was addressed, further scaling naturally showed diminishing returns.

To validate our claims, we performed additional experiments, reducing the dataset from 100M to 50M while keeping training iterations constant. By scaling the model size to 596M and 1B, we surpassed the results of the method ZbWb referenced, proving that our improvements were unrelated to data scaling. This highlights ZbWb’s fundamental misunderstanding of our work.  He/she has factual errors.

-----

### **3. Misguided Comparisons with Unreasonable Expectations**

ZbWb criticized our comparisons, we found that the comparison papers (methods) provided [4] are entirely non-comparable.

Our paper evaluates adversarial training using synthetic data generated with diffusion models, focusing exclusively on ImageNet-1K. Our comparisons are intentionally designed against methods also restricted to ImageNet-1K, which is fair and appropriate.

Fair comparison requires using the same real dataset, a basic principle followed by all relevant top-conference papers (e.g., [1]\[2]\[3]) on RobustBench [1]; However, ZbWb requested that we compare against methods that use DataComp-1B, a vastly larger external dataset.

Furthermore, ZbWb also ignored many other critical factors of that paper:

- Uses 5x larger models
- Processes 7x the total data iterations
- Requires *thousands of TPU days* for training (compared to the ~200 TPU days used in our work).
- Even failed to provide open-source GitHub code.

In response, we scaled up two models, using 300+700 TPU-v4 days, a massive increase in resources. For many researchers without access to such computational budgets, replicating this would be impossible. We believe ZbWb’s expectation for comparisons with such resource-intensive methods is inappropriate and unreasonably burdensome for rebuttal or follow-up work.

------


### **4. Misplaced Focus on CFG Tuning**

ZbWb heavily criticized us for not performing exhaustive CFG (classifier-free guidance) parameter sweeps, labeling this as a major weakness. In fact, not relying on tuning CFG is one of our advantages.

Like another reviewer, UWWE, recognized that one of our contributions is that our simple yet effective approach addresses the distribution alignment issue, achieving SOTA performance without the need for extensive CFG tuning.

Also, ZbWb acknowledged that sweeping CFG values is computationally intensive, requiring thousands of TPU days (more than 400M synthetic data generation), far exceeding the entire compute budget of our paper (~900 TPU days for 100M synthetic data generation and <200 TPU days for training). It is both impractical and irrelevant to our contributions. We consider this criticism unprofessional and entirely unhelpful.

------

### 5. Reviewer ZbWb's Excessive Factual Errors Have Left Us With No Space to Highlight Our Novelty  

Due to the overwhelming number of factual inaccuracies in ZbWb's review, our rebuttal was largely forced to focus on correcting these errors instead of reinforcing the key contributions and novelty of our work. We want to take this opportunity to emphasize our core contribution.  

Our work solves a four-year-long challenge in the adversarial robustness community. Since [5] attempted to use Diffusion Models for adversarial training in ImageNet (2021), the performance improvement was marginal, and later non-diffusion-based methods outperformed it. We are the first to unlock the true potential of Diffusion Models in adversarial robustness, demonstrating a clear advantage over non-diffusion-based synthetic data methods on ImageNet. This contribution was recognized by other reviewers, including 8STQ, who explicitly listed it as a key strength, yet ZbWb deliberately ignored it.  

ZbWb also misrepresented our work, incorrectly equating our method to [4] simply because both involve a two-stage approach. However, [4] does not use synthetic data at all, making the comparison fundamentally flawed. Moreover, the starting point of both approaches is entirely different, further highlighting the inaccuracy of the comparison. This deliberate misinterpretation significantly misrepresents our novelty and contribution.


### Additional Issues with Reviewer ZbWb's Assessment  

1. **Excessive Speculation and Misinterpretation**  
   Reviewer ZbWb made numerous baseless assumptions about our work, which we find highly inappropriate. Nowhere in our paper do we claim to perform "data scaling." However, ZbWb incorrectly assumes that our method is based on "data scaling," equating it to [4] and criticizing us for a lack of technical contribution. We clearly refuted this in our rebuttal: our method does **not** perform data scaling. In fact, we demonstrated that even when reducing the dataset size from 100M to 50M, our method still significantly outperforms the supposed "data scaling" state-of-the-art [4], directly contradicting ZbWb's claim.  

2. **Incorrect Assumption About Saturated Performance**  
   ZbWb claims that our model's performance is saturated. However, this misunderstanding arises because we increased the dataset size without increasing the number of data iterations. The observed performance improvement stems from alleviating overfitting by increasing the data size, not from reaching a saturation point. Once overfitting is mitigated, further increasing the dataset size naturally does not yield additional gains. ZbWb's assumption that we are performance-saturated is thus incorrect.  

3. **Fundamental Misunderstanding of Robustness Concepts**  
   ZbWb argues that using generated data to improve robustness on an OOD dataset was already shown in [6], and therefore, our work lacks novelty. However, this claim is flawed for multiple reasons:  
   - **[6] does not use a two-stage approach**, making the comparison inappropriate.  
   - The "robustness" discussed in [6] is **not** "adversarial robustness." This is a fundamentally different concept.  
   - In both ImageNet [1-5] and CIFAR [7-8], all prior works on adversarial robustness rely on adversarial training. Other types of robustness improvements do not transfer to adversarial robustness.  
   - ZbWb wrongly assumes that because a method improves OOD robustness, it should automatically improve adversarial robustness, then our work lacks novelty. This is a **severe logical fallacy**—akin to assuming that because someone can drive a car, they must also be able to fly a plane.  

These misinterpretations and errors severely misrepresent our work and its contributions. We urge the committee to recognize these issues and evaluate our paper based on its actual merits rather than Reviewer ZbWb’s factually incorrect claims.  


------

### **Hope**

Given the above points,  according to the academic spirit and objective principles of CVPR,  we hope that you investigate the quality and validity of Reviewer ZbWb’s review. Specifically:

1. **Determine whether LLM assistance was used in generating the review.** The factual errors, lack of depth, and inconsistency with the review quality expected at CVPR strongly suggest this possibility.
2. **Assess the fairness and professionalism of the review.** ZbWb’s unreasonable demands and misrepresentations of our work have placed an undue burden on us and demonstrate a lack of responsibility in evaluating our submission.

We are confident that a closer examination will reveal significant flaws in the review, and we hope you will take appropriate steps to ensure the review process remains fair and rigorous.

Thank you for your time and consideration.


[1] Croce F, Andriushchenko M, Sehwag V, et al. Robustbench: a standardized adversarial robustness benchmark[J]. arXiv preprint arXiv:2010.09670, 2020.

[2] Liu C, Dong Y, Xiang W, et al. A comprehensive study on robustness of image classification models: Benchmarking and rethinking[J]. International Journal of Computer Vision, 2024: 1-23.

[3] Singh N D, Croce F, Hein M. Revisiting adversarial training for imagenet: Architectures, training and generalization across threat models[J]. Advances in Neural Information Processing Systems, 2024, 36.

[4] Wang Z, Li X, Zhu H, et al. Revisiting Adversarial Training at Scale[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 24675-24685.

[5] Gowal et al., "Improving Robustness using Generated Data", 2021

[6] Bansal, Hritik, et al. "Leaving reality to imagination: Robust classification via generated datasets." ICLR Workshop on Trustworthy and Reliable Large-Scale Machine Learning Models (2023).

[7] Bartoldson B R, Diffenderfer J, Parasyris K, et al. Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies[J]. ICML, 2024.

[8] Wang, Zekai, et al. "Better diffusion models further improve adversarial training." International Conference on Machine Learning. PMLR, 2023.

Sincerely,

Submission 16072 Authors"""




def test_qwen2_fast_jit():
    dtype = jnp.bfloat16
    # dtype = jnp.float32
    max_cache_length=8192

    model_path = 'Qwen/Qwen2-7B-Instruct'
    model_path= 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

    # jax.config.update('jax_platform_name', 'cpu')
    # variables=model.init(rng,x.astype(jnp.int32),x.astype(jnp.int32))
    # params=variables['params']

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
        AutoTokenizer, AutoModelForCausalLM,AutoConfig,Qwen2Tokenizer


    llama_config=Qwen2Config()
    print(llama_config)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(config)
    # config.init_cache=llama_config.init_cache
    cache = init_cache(config,1,max_cache_length=max_cache_length, dtype=dtype)

    # while True:
    #     pass

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # exit_token_ids = 151645
    exit_token_ids=tokenizer.eos_token_id

    print(f'{tokenizer.eos_token=} ,{tokenizer.eos_token_id=}, {exit_token_ids=}')

    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )

    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "system", "content": ""},
        # {"role": "user", "content": "Who are you?"},
        # {"role": "user", "content": "Give me a short introduction to large language model."},
        # {"role": "user", "content": "1+1=2 1+2=?"},
        # {"role": "user", "content": "Mnist examples in jax?"},
        {"role": "user", "content": content,
                                    },
        #
        # {"role": "assistant", "content": "A large language model is a type of artificial intelligence (AI) model that"},
        # {"role": "assistant", "content": "A large language model is a type of"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
                                           # add_generation_prompt=False,continue_final_message=True)
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="jax",padding='longest',max_length=max_cache_length)

    input_ids = inputs['input_ids']


    l=input_ids.shape[1]

    attention_mask = inputs['attention_mask']
    pad_attention=jnp.zeros_like(attention_mask).at[:,0].set(1)

    input_ids_pad=jnp.pad(input_ids, ((0, 0), (0, max_cache_length - attention_mask.sum(axis=1)[0])),constant_values=tokenizer.eos_token_id)
    pad_attention = jnp.pad(pad_attention, ((0, 0), (0, max_cache_length - attention_mask.sum(axis=1)[0])))



    # print(pad_attention)
    # print(input_ids_pad[0,:64])
    # while True:
    #     pass


    state_dict = model.state_dict()
    params = convert_torch_to_flax_llama(state_dict)

    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)
    position_ids = jnp.arange(0, 1)[None, ...]

    model = Qwen2ForCausalLM(config)
    jit_infer = jax.jit(model.apply)

    temperature=0.6

    res = []
    print('go')
    key=jax.random.PRNGKey(1)
    for i in tqdm(range(max_cache_length)):
        """
        """
        # input_ids = select_ids[..., None]
        # print(input_ids_pad[:,i:i+1].shape)

        logits, cache = jit_infer({'params': params}, input_ids=input_ids_pad[:,i:i+1], position_ids=position_ids,
                                  attention_mask=pad_attention, cache=cache)


        logits/=temperature
        key,key2=jax.random.split(key)

        def sample_fn(logits_batch):
            # print(logits_batch.shape)
            return  jax.random.choice(key2,jnp.arange(0,logits_batch.shape[0],dtype=jnp.int32),axis=0,p=flax.linen.softmax(logits_batch))


        select_ids=jax.vmap(sample_fn)(logits[:,-1])

        # select_ids=jax.random.choice(key2,logits[:, -1],axis=1,p=flax.linen.softmax(logits[:,-1]))

        # select_ids = jnp.argmax(logits[:, -1], axis=1)

        if i<l-1:
            select_ids=input_ids_pad[:,i+1]
        else:
            pass
        # select_ids=jnp.where(input_ids_pad.at[:,i+1]!=tokenizer.eos_token_id  ,input_ids_pad[:,i+1] ,select_ids    )

        # res.append(select_ids)
        res.append(select_ids.astype(jnp.int32))
        if select_ids[0] == exit_token_ids :
            break
        position_ids += 1
        pad_attention = pad_attention.at[:,  i+1].set(1)
        input_ids_pad=input_ids_pad.at[:,i+1].set(select_ids)



        if (i+1)%500==0:
            output = \
                tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)[
                    0]
            # output=tokenizer.decode(np.array(res), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print('\n'*5)
            print(output)



        # print(tokenizer.decode(select_ids))

    # ans = []
    # for t in res:
    #     decoded_token = tokenizer.decode(t)
    #     print(decoded_token, end='', flush=True)
    #     ans.append(decoded_token)
    # print('\n' * 10)
    #
    # print(np.array(res))
    output = \
        tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)[
            0]
    # output=tokenizer.decode(np.array(res), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output)





def test_qwen2():
    dtype = jnp.bfloat16
    # dtype = jnp.float32
    max_cache_length=2048

    model_path = 'Qwen/Qwen2-7B-Instruct'
    model_path= 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

    # jax.config.update('jax_platform_name', 'cpu')
    # variables=model.init(rng,x.astype(jnp.int32),x.astype(jnp.int32))
    # params=variables['params']

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
        AutoTokenizer, AutoModelForCausalLM,AutoConfig,Qwen2Tokenizer


    llama_config=Qwen2Config()
    print(llama_config)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(config)
    # config.init_cache=llama_config.init_cache
    cache = init_cache(config,1,max_cache_length=max_cache_length, dtype=dtype)

    # while True:
    #     pass

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # exit_token_ids = 151645
    exit_token_ids=tokenizer.eos_token_id

    print(f'{tokenizer.eos_token=} ,{tokenizer.eos_token_id=}, {exit_token_ids=}')

    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )

    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "system", "content": ""},
        # {"role": "user", "content": "Who are you?"},
        # {"role": "user", "content": "Give me a short introduction to large language model."},
        {"role": "user", "content": "1+1=2 1+2=?"},
        #
        # {"role": "assistant", "content": "A large language model is a type of artificial intelligence (AI) model that"},
        # {"role": "assistant", "content": "A large language model is a type of"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
                                           # add_generation_prompt=False,continue_final_message=True)
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="jax")


    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    state_dict = model.state_dict()
    params = convert_torch_to_flax_llama(state_dict)

    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x), dtype=dtype), params)

    pad_attention = jnp.pad(attention_mask, ((0, 0), (0, max_cache_length - attention_mask.sum(axis=1)[0])))

    input_ids = input_ids
    position_ids = jnp.arange(0, input_ids.shape[1])[None, ...]

    # llama_config = LlamaConfig(max_cache_length=1024)

    model = Qwen2ForCausalLM(config)

    b, n, d = shape = 1, 1, 768

    jit_infer = jax.jit(model.apply)

    logits, cache = jit_infer({'params': params}, input_ids=input_ids, position_ids=position_ids,
                              attention_mask=pad_attention, cache=cache)
    # select_ids = jnp.argmax(logits[:, -1], axis=1)
    # decoded_token = tokenizer.decode(select_ids)
    # print(decoded_token, end='', flush=True)

    # print(out)

    position_ids = position_ids[:, -1][..., None]
    max_decode_length = max_cache_length
    res = []
    # exit_token_ids=tokenizer.encode(tokenizer.st)[0]


    # while True:
    #     params


    for i in tqdm(range(max_decode_length)):
        select_ids = jnp.argmax(logits[:, -1], axis=1)
        # print(select_ids)
        res.append(select_ids)

        if select_ids[0] == exit_token_ids:
            break

        input_ids = select_ids[..., None]
        position_ids += 1
        pad_attention = pad_attention.at[:, attention_mask.sum(axis=1)[0] + i].set(1)
        logits, cache = jit_infer({'params': params}, input_ids=input_ids, position_ids=position_ids,
                                  attention_mask=pad_attention, cache=cache)

        # print(tokenizer.decode(select_ids))

    # ans = []
    # for t in res:
    #     decoded_token = tokenizer.decode(t)
    #     print(decoded_token, end='', flush=True)
    #     ans.append(decoded_token)
    # print('\n' * 10)
    #
    # print(np.array(res))
    output = \
        tokenizer.batch_decode(np.array(res).reshape(1, -1), skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)[
            0]
    # output=tokenizer.decode(np.array(res), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output)



def test_qwen_torch():
    dtype = jnp.bfloat16
    # model_path = 'Qwen/Qwen2-7B-Instruct'
    model_path = '/autodl-tmp/qwen/Qwen2-7B-Instruct'
    # model_path = './autodl-tmp/qwen/Qwen2-7B-Instruct'

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizerFast, \
        AutoTokenizer, AutoModelForCausalLM,AutoConfig,Qwen2Tokenizer


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,)
    from transformers.generation import configuration_utils

    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the base model with adapters on top
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )


    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": "Who are you?"},
        {"role": "user", "content": "Give me a short introduction to large language model."},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt)


    device = 'cuda'
    # device = 'cpu'
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out=model.generate(**inputs,
                       do_sample=False,
                       # temperature=0.1,
                       # top_k=1000,
                       # top_p=1.0,
                       max_new_tokens=512)
    print(out)
    output=tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
    #
    #
    # print('\n'*10)
    # while True:


if __name__=="__main__":
    # test_qwen_torch()
    test_qwen2_fast_jit()
    # test_qwen2()

    # cache=test_vicuna("I'm a")
    """
    cache2 = test_vicuna("I'm a language model called")

    layer_name = f'layer_{28}'
    layer_cache = cache[layer_name]['v'][0,0,55:88,0]
    layer_cache2 = cache2[layer_name]['v'][0,0,55:88,0]

    #(batch_size, num_heads, cache_size, head_dim)
    print(layer_cache)
    print('\n'*2)
    print(layer_cache2)

    jnp.allclose(layer_cache,layer_cache2)
    print(layer_cache-layer_cache2)
    print(jnp.sum(jnp.abs(layer_cache-layer_cache2)))

    """









    # test_vicuna_torch()