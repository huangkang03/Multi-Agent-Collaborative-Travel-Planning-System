"""
模型客户端的统一入口 -- 多智能体系统里所有 agent 共用的模型配置。

各模型名的单一来源（换模型只改一处）：
- MODEL     qwen3.8-flash，轻量快速，用于对话与结构化抽取（analyst）
- KIMI_MODEL kimi-k3，深度思考模型，用于需要推理决策的环节（api_selector）

create_*_client() 为调用方创建独立的 OpenAI 客户端实例，
各 agent 各自持有、互不共享。密钥和服务地址从 .env 读取。
"""

import os

import dotenv
from openai import OpenAI

# 阿里云百炼的模型名
MODEL = "qwen3.8-flash"
KIMI_MODEL = "kimi-k3"


def _load_env() -> None:
    dotenv.load_dotenv()


def create_client() -> OpenAI:
    """创建一个 qwen 客户端（对话/抽取用）。"""
    _load_env()
    return OpenAI(
        api_key=os.getenv("ALIBABA_qwenflash_API_KEY"),
        base_url=os.getenv("ALIBABA_qwenflash_url"),
    )


def create_kimi_client() -> OpenAI:
    """创建一个 kimi-k3 客户端（思考决策用）。"""
    _load_env()
    return OpenAI(
        api_key=os.getenv("kimi_k3_api_key"),
        base_url=os.getenv("kimi_k3_url"),
    )
