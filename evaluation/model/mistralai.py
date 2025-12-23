import os
from time import sleep
from vllm import LLM, SamplingParams


def get_model(model_name, api_key):
    tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))

    return LLM(
        model=model_name,
        dtype="float16",
        max_model_len=16384,
        tensor_parallel_size=tp_size,
        trust_remote_code=False,  # Mistral 不需要
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
            outputs = model.chat(messages, sampling_params=params)
            return outputs[0].outputs[0].text
        except Exception as e:
            print(f"[Mistral vLLM] {e}")
            sleep(10)
