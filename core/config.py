# core/config.py
import os
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logging.basicConfig(level=logging.WARNING, force=True)
for _n in ("httpx", "httpcore", "openai", "openai._base_client"):
    logging.getLogger(_n).setLevel(logging.ERROR)

def get_llm(temperature: float = 0.0):
    model_name = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
    local_url = os.getenv("LOCAL_API_BASE", "http://localhost:11434/v1")
    print(f"🚀 正在连接本地 Ollama 接口: {model_name} (at {local_url}) ...")
    return ChatOpenAI(
        model=model_name,
        api_key="ollama",
        base_url=local_url,
        temperature=temperature,
    )

def get_embeddings():
    embed_model = os.getenv("LOCAL_EMBED_MODEL", "mxbai-embed-large")
    local_url = os.getenv("LOCAL_API_BASE", "http://localhost:11434/v1")
    print(f"🧠 正在加载本地 Embeddings: {embed_model} ...")
    return OpenAIEmbeddings(
        model=embed_model,
        api_key="ollama",
        base_url=local_url,
    )

# ---- Calibration hyperparameters (A–F) ----
CALIB_TW = int(os.getenv("GOGOBOT_CALIB_TW", "5"))          # short window
CALIB_TM = int(os.getenv("GOGOBOT_CALIB_TM", "30"))         # long window (tm > tw)
CALIB_J = int(os.getenv("GOGOBOT_CALIB_J", "3"))            # directions
CALIB_EVERY_N_TURNS = int(os.getenv("GOGOBOT_CALIB_EVERY", "2"))  # trigger frequency

try:
    llm = get_llm(temperature=0.0)
    embeddings = get_embeddings()
except Exception as e:
    print(f"❌ 本地模型配置错误: {e}")
    print("💡 请确保已安装 Ollama 且已通过 `ollama run qwen2.5:7b` 启动模型")
    raise
