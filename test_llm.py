"""
# test_llm.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 加载 .env 文件中的 API Key
load_dotenv()

# 检查 Key 是否存在
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ 错误: 未找到 OPENAI_API_KEY，请检查您的 .env 文件。")
    exit(1)
else:
    print(f"✅ 成功读取 API Key: {api_key[:8]}******")

# 2. 初始化大模型
try:
    print("正在连接 OpenAI...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # 3. 发送简单请求
    response = llm.invoke("你好，请做一个简单的自我介绍，并告诉我新加坡在哪里。")
    
    print("\n" + "="*20 + " 模型回复 " + "="*20 + "\n")
    print(response.content)
    print("\n" + "="*50)
    print("🎉 测试成功！大模型连接正常。")

except Exception as e:
    print(f"\n❌ 连接失败: {e}")
    print("请检查您的网络连接（国内可能需要代理）或 Key 是否有效。")
"""

# core/config.py
import os
import logging
# 虽然用的是 Ollama，但因为其兼容 OpenAI 接口，使用 ChatOpenAI 类是最稳定、功能支持最全的方式（支持 Tool Calling）
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- 1. 配置日志 (屏蔽第三方库的噪音) ---
logging.basicConfig(level=logging.WARNING, force=True)
for _n in ("httpx", "httpcore", "openai", "openai._base_client"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# --- 2. 配置 LLM (大模型) ---
def get_llm(temperature=0.5):
    """
    获取 LLM 实例，连接本地 Ollama 服务
    """
    # 这里必须与您在终端 'ollama run' 下载的模型名称一致
    # M4 芯片如果内存够大(16G+)，也可以尝试改用 "qwen2.5:14b"
    model_name = "qwen2.5:7b" 
    
    print(f"🔌 正在连接本地模型: {model_name} @ localhost:11434 ...")
    
    return ChatOpenAI(
        model=model_name,
        # 指向本地 Ollama 的 API 地址
        base_url="http://localhost:11434/v1",
        # Ollama 不验证 Key，但 LangChain 客户端要求必填，随便填即可
        api_key="ollama", 
        temperature=temperature
    )

# --- 3. 配置 Embeddings (向量模型) ---
def get_embeddings():
    """
    获取 Embedding 实例，连接本地 Ollama
    用于 Wikipedia 检索等功能
    """
    # 必须先在终端运行过: ollama pull nomic-embed-text
    return OpenAIEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        # 显式关闭上下文长度检查，避免本地模型因 metadata 缺失报错
        check_embedding_ctx_length=False 
    )

# --- 4. 初始化全局实例 ---
try:
    llm = get_llm()
    embeddings = get_embeddings()
except Exception as e:
    print(f"❌ 本地模型连接配置错误: {e}")
    print("请检查：\n1. Ollama 是否已启动 (顶部菜单栏是否有图标)\n2. 是否已下载模型 (ollama run qwen2.5:7b)")
    raise e
