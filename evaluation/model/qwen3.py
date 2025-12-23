import os
from time import sleep
from vllm import LLM, SamplingParams


def get_model(model_name, api_key):
    tensor_parallel_size = int(os.environ.get("VLLM_TP_SIZE", "1"))

    return LLM(
        model=model_name,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=32000,
        tensor_parallel_size=tensor_parallel_size,
    )


def generate(model, model_name, system_prompt, user_prompt, temperature):
    params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.1,
        top_p=0.9,
        max_tokens=4096,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    while True:
        try:
            out = model.chat(messages, sampling_params=params)
            return out[0].outputs[0].text
        except Exception as e:
            print(f"[Qwen3 vLLM] error: {e}")
            sleep(10)
