import os
import json
from time import sleep
from vllm import LLM, SamplingParams


def _read_max_len_from_config(model_path: str):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # 常见字段优先级：model_max_length > max_position_embeddings
    max_len = cfg.get("model_max_length", None)
    if max_len is None:
        max_len = cfg.get("max_position_embeddings", None)

    if max_len is None:
        return None

    # 有些 config 里会是 4096.0
    try:
        return int(max_len)
    except Exception:
        return None


def get_model(model_name, api_key=None):
    tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
    max_len = _read_max_len_from_config(model_name)

    kwargs = dict(
        model=model_name,
        dtype="float16",
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
    )

    # 只有读到了长度才传，避免传 None 触发校验问题
    if max_len is not None:
        kwargs["max_model_len"] = max_len

    # 有些 vLLM 版本不接受 disable_log_stats（你之前日志里是通过 args 注入的）
    # 如果你确定你这版能用，可以加回去；否则先别传，最稳。
    # kwargs["disable_log_stats"] = True

    return LLM(**kwargs)


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
            print(f"[deepseek vLLM] error: {e}")
            sleep(10)
