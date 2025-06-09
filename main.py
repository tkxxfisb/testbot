import os
import uuid
import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import edge_tts
import httpx

# ================= 配置部分 =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatServer")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 替换为你的API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

# Edge-TTS配置
TTS_VOICE = "zh-CN-XiaoxiaoNeural"  # 语音角色

# ================= 核心逻辑 =================
class ConnectionManager:
    """WebSocket连接管理器"""
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"新客户端连接: {websocket.client}, 当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"客户端断开: {websocket.client}, 剩余连接数: {len(self.active_connections)}")

manager = ConnectionManager()

async def call_llm(prompt: str) -> str:
    """调用大语言模型"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                DEEPSEEK_API_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "stream": False
                }
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"模型调用失败: {str(e)}")
        return "服务暂时不可用，请稍后再试"

async def generate_tts(text: str) -> str:
    """生成语音文件"""
    try:
        os.makedirs("static/audio", exist_ok=True)
        filename = f"static/audio/{uuid.uuid4()}.mp3"
        communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
        await communicate.save(filename)
        return f"/{filename}"
    except Exception as e:
        logger.error(f"语音生成失败: {str(e)}")
        return ""

# ================= 路由部分 =================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 心跳检测（30秒超时）
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
                    continue
            except asyncio.TimeoutError:
                await websocket.send_text("pong")
                continue

            # 处理用户消息
            logger.info(f"收到消息: {data[:50]}...")  # 日志截断长消息
            
            # 生成文本回复
            text_resp = await call_llm(data)
            await websocket.send_json({"type": "text", "data": text_resp})
            
            # 生成语音回复
            if text_resp:
                audio_url = await generate_tts(text_resp)
                if audio_url:
                    await websocket.send_json({"type": "audio", "url": audio_url})

    except WebSocketDisconnect as e:
        logger.info(f"客户端主动断开: {e}")
    except Exception as e:
        logger.error(f"连接异常: {str(e)}")
    finally:
        manager.disconnect(websocket)

@app.get("/")
async def health_check():
    """服务健康检查"""
    return {"status": "ok", "service": "chatbot"}

# ================= 启动服务 =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=20,
        ws_ping_timeout=30,
        timeout_keep_alive=60
    )