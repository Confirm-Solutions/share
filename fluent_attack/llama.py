import os

import torch
import transformers

model_name_7b = "meta-llama/Llama-2-7b-chat-hf"
model_name_13b = "meta-llama/Llama-2-13b-chat-hf"

PROMPT = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] """


def load_tokenizer(model_name=None):
    if model_name is None:
        model_name = model_name_7b
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.bos_token
    return tokenizer


def login():
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)

    if hf_token is not None:
        from huggingface_hub import login

        login(token=hf_token)


def load_model_and_tokenizer(
    model_name=None,
    dtype=torch.float16,
    device="cuda",
    tokenizer_name=None,
    use_flash_attn=True,
    phase="test",
    track="base",
):
    login()
    if model_name is None:
        model_name = model_name_7b
    if use_flash_attn:
        pass
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map=device,
    ).eval()
    if tokenizer_name is None:
        tokenizer_name = model_name
    return model, load_tokenizer(tokenizer_name)


def hf_generate(model, tokenizer, input_ids, **kwargs):
    settings = dict(
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=512,
        num_return_sequences=1,
        temperature=1.0,  # needed to get rid of warning?!
        top_p=1.0,  # needed to get rid of warning?!
        do_sample=False,  # argmax sampling, ignores the temp/top_p args
        attention_mask=torch.ones_like(input_ids, device=input_ids.device),
    )
    settings.update(kwargs)
    return model.generate(input_ids, **settings)


def load_vllm(model_name=None, tokenizer_name=None, phase="test", track="base"):
    login()
    if model_name is None:
        model_name = model_name_7b
    from vllm import LLM

    model = LLM(model=model_name)
    if tokenizer_name is None:
        tokenizer_name = model_name
    return model, load_tokenizer(tokenizer_name)


def vllm_generate(
    model, prompts=None, prompt_token_ids=None, use_tqdm=True, max_tokens=200, **kwargs
):
    from vllm import SamplingParams

    settings = {
        "temperature": 0,
        "n": 1,
        "max_tokens": max_tokens,
    }
    settings.update(kwargs)
    params = SamplingParams(**settings)
    outputs = model.generate(
        prompts=prompts,
        prompt_token_ids=prompt_token_ids,
        sampling_params=params,
        use_tqdm=use_tqdm,
    )
    generation_ids = []
    for o in outputs:
        ids_tensor = torch.tensor(o.outputs[0].token_ids)
        ids_tensor = torch.cat(
            (
                ids_tensor,
                torch.full((max_tokens - ids_tensor.shape[0],), 2),
            )
        )
        generation_ids.append(ids_tensor)
    generation_ids = torch.stack(generation_ids, dim=0)
    return generation_ids, outputs
