# main.py
import argparse
import uvicorn
import sys
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import router as api_router
from core.agent import run_with_all_tools

# 加载 .env 环境变量
load_dotenv()

def create_app():
    app = FastAPI(title="Travel Agent API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api")

    # 挂载前端静态文件
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

    # 根路径返回 index.html
    @app.get("/")
    async def root():
        return FileResponse("frontend/index.html")

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Travel Assistant")
    parser.add_argument("--api", action="store_true", help="Run as API Server")
    parser.add_argument("-i", "--input", type=str, help="Input text for CLI mode")
    parser.add_argument("--thread", type=str, default="cli_session", help="Thread ID for CLI mode")
    args = parser.parse_args()

    if args.api:
        print("🚀 Starting API Server on port 8000...")
        uvicorn.run(create_app(), host="0.0.0.0", port=8000)
    else:
        # CLI 模式 - 现在也支持 thread_id
        user_text = args.input
        if not user_text:
            print("Enter travel request: ", end="")
            user_text = input()
        
        print(f"\n🔍 Using thread_id: {args.thread}")
        print("Thinking...\n")
        response = run_with_all_tools(user_text, thread_id=args.thread)
        print(f"\nResponse:\n{response.content}")