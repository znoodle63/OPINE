from dataclasses import dataclass

from openai import OpenAI

from scoring.prompts import SYSTEM_PROMPT_OPINION


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2


def build_llm_client(cfg: LLMConfig) -> OpenAI:
    client = OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
    )
    return client


def call_llm(client: OpenAI, cfg: LLMConfig, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_OPINION},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
    )
    return resp.choices[0].message.content or ""
