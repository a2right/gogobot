# app/api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os
import uuid
from core.agent import run_with_all_tools

router = APIRouter()


class ChatRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None  # 改为可选


class ChatResponse(BaseModel):
    response: str
    thread_id: str  # 总是返回 thread_id


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 如果前端没有提供 thread_id，后端自动生成一个新的
        if not request.thread_id or request.thread_id == "default":
            thread_id = str(uuid.uuid4())
            print(f"🆕 Generated new thread_id: {thread_id}")
        else:
            thread_id = request.thread_id
            print(f"🔍 Using existing thread_id: {thread_id}")
        
        response = run_with_all_tools(
            user_input=request.user_input,
            thread_id=thread_id
        )
        
        return ChatResponse(
            response=response.content,
            thread_id=thread_id  # 返回给前端
        )
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations")
async def list_conversations() -> List[Dict]:
    """列出所有会话"""
    db_file = os.environ.get("GOGOBOT_DB_FILE", "chat_db.json")
    if not os.path.exists(db_file):
        return []
    
    try:
        with open(db_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    
    conversations = []
    for thread_id, container in data.items():
        if not isinstance(container, dict):
            continue
        history = container.get("history", [])
        if history and isinstance(history, list):
            first_msg = next(
                (msg.get("content", "")[:50] for msg in history if msg.get("role") == "user"),
                "New Conversation"
            )
            conversations.append({
                "thread_id": thread_id,
                "title": first_msg,
                "message_count": len(history)
            })
    
    return conversations


@router.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    """删除指定会话"""
    db_file = os.environ.get("GOGOBOT_DB_FILE", "chat_db.json")
    if not os.path.exists(db_file):
        return {"success": False, "message": "Database file not found"}
    
    try:
        with open(db_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if thread_id in data:
            del data[thread_id]
            tmp = db_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, db_file)
            return {"success": True, "message": f"Deleted conversation {thread_id}"}
        
        return {"success": False, "message": "Conversation not found"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.post("/conversations/new")
async def create_new_conversation():
    """生成一个新的 conversation_id（可选接口，前端也可以自己生成）"""
    new_id = str(uuid.uuid4())
    return {"thread_id": new_id}