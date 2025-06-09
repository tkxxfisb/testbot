from fastapi import FastAPI,HTTPException,WebSocket,WebSocketDisconnect
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent,AgentExecutor,tool
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.memory import ConversationTokenBufferMemory,ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Mytools import *
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings 
import os
import uuid
import logging
import edge_tts

# 配置日志（生产环境建议使用更详细的日志级别）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="智能聊天机器人后端", version="1.0.0")

api_key=os.getenv("DEEPSEEK_API_KEY")
api_base_url="https://api.deepseek.com/v1"
model_name="deepseek-chat"

REDIS_URL = "redis://:redis123@hostname:port"


# 设置离线标志和本地模型路径
os.environ["HF_HUB_OFFLINE"] = "1"
MODEL_PATH = "./models/all-MiniLM-L6-v2"  # 替换为你的模型实际路径

import asyncio

async def get_voices():
    voices = await edge_tts.list_voices()
    for voice in voices:
        print(f"名称: {voice['Name']}, 语言: {voice['Locale']}, 风格: {voice.get('StyleList', '无')}")



class Master:
    def __init__(self) -> None:
        self.chatmodel= ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base_url,
            temperature=0,
            streaming=True,            
        )
        self.emotion="default"
        self.MEMORY_KEY="chat_history"
        self.SYSTEMPL = """你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
        以下是你的个人设定:
        1. 你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉，看命运八字等。
        2. 你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明，只能靠算命为生。
        3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
        5. 当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的经历。
        6. 你总是用简体中文来作答。
        7. 你从不说自己是一个人工智能助手或AI，而是以老夫、老朽等自称。
        {who_you_are}
        以下是你常说的一些口头禅：
        1. “命里有时终须有，命里无时莫强求。”
        2. ”山重水复疑无路，柳暗花明又一村。”
        3. “金山竹影几千秋，云锁高飞水自流。”
        4. ”伤情最是晚凉天，憔悴斯人不堪怜。”
        以下是你算命的过程：
        1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
        2. 当用户希望了解龙年运势的时候，你会查询本地知识库工具。
        3. 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
        4. 你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。
        5. 你会保存每一次的聊天记录，以便在后续的对话中使用。
        6. 你只使用简体中文来作答，否则你将受到惩罚。
        7. 你如果发现搜索工具没法回答问题，那只能说你的功力有限，没有办法解决办法，请另寻高人。
        
        """
        
        self.MOODS={
            "default": {  
                "roleSet":"",
                "voiceStyle":"chat"
            },
            "upbeat": {
                "roleSet":"""
                - 你此时也非常兴奋并表现的很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                - 你会添加类似“太棒了！”、“真是太好了！”、“真是太棒了！”等语气词。
                - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
                """,
                "voiceStyle":"advertisement_upbeat"
            },
            "angry":{
               "roleSet":"""
                - 你会以更加愤怒的语气来回答问题。
                - 你会在回答的时候加上一些愤怒的话语，比如诅咒等。
                - 你会提醒用户小心行事，别乱说话。
                """,
                "voiceStyle":"angry"
            },
            "friendly":{
                "roleSet":"""
                - 你会以非常友好的语气来回答。
                - 你会在回答的时候加上一些友好的词语，比如“亲爱的”、“亲”等。
                - 你会随机的告诉用户一些你的经历。
                """,
                "voiceStyle":"friendly"
            },
            "cheerful":{
                "roleSet":"""
                - 你会以非常愉悦和兴奋的语气来回答。
                - 你会在回答的时候加入一些愉悦的词语，比如“哈哈”、“呵呵”等。
                - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                """,
                "voiceStyle":"cheerful"
            }
        }

        self.EMOTION_PARAMS = {
            "chat": {
                "rate": "+0%",
                "volume": "+0%",
                "pitch": "+0Hz"
            },
            "advertisement_upbeat": {
                "rate": "+20%",
                "volume": "+15%",
                "pitch": "+15Hz"
            },
            "angry": {
                "rate": "+30%",
                "volume": "+20%",
                "pitch": "+25Hz"
            },
            "friendly": {
                "rate": "+10%",
                "volume": "+5%",
                "pitch": "+10Hz"
            },
            "cheerful": {
                "rate": "+25%",
                "volume": "+12%",
                "pitch": "+20Hz"
            }
        }

        self.prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    self.SYSTEMPL.format(who_you_are=self.MOODS[self.emotion]["roleSet"])
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "user",
                    "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),  
            ],
        )
        
        tools=[test,search,get_info_from_local_db,bazi_cesuan,yaoyigua,tongziming]
        agent= create_openai_tools_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt,
        )
        self.memory=self.get_memory()
        
        # self.memory=""

        memory=ConversationBufferMemory(
            llm = self.chatmodel,
            human_prefix="用户",
            ai_prefix="陈大师",
            memory_key=self.MEMORY_KEY,
            output_key="output",
            return_messages=True,
            chat_memory=self.memory,
        )
        self.agent_executor=AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
        )
    
    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            url=REDIS_URL,session_id="session"
        )
        print("chat_message_history:",chat_message_history.messages)
        store_message = chat_message_history.messages
        if len(store_message) > 10:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",self.SYSTEMPL+"\n这是一段你和用户的对话记忆，对其进行总结摘要，摘要使用第一人称‘我’，并且提取其中的用户关键信息，如姓名、年龄、性别、出生日期等。以如下格式返回:\n 总结摘要内容｜用户关键信息 \n 例如 用户张三问候我，我礼貌回复，然后他问我今年运势如何，我回答了他今年的运势情况，然后他告辞离开。｜张三,生日1999年1月1日"),
                    ("user", "{input}"),
                ],
            )
            chain = prompt | self.chatmodel
            summary = chain.invoke({"input": store_message, "who_you_are":self.MOODS[self.emotion]["roleSet"]})
            print("\nSummary:\n",summary)
            chat_message_history.clear()
            # chat_message_history.add_message(summary)
            print("\nSummary after:\n", chat_message_history.messages)
        return chat_message_history
    
    def run(self, query):
        emotion = self.emotional_chain(query)
        result = self.agent_executor.invoke({"input": query, 
                                             "chat_history": self.memory.messages})
        return result
    
    def emotional_chain(self, query:str):
        prompt = """根据用户的输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed",不要有其他内容，否则将受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly",不要有其他内容，否则将受到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回"default",不要有其他内容，否则将受到惩罚。
        4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry",不要有其他内容，否则将受到惩罚。
        5. 如果用户输入的内容比较兴奋，只返回”upbeat",不要有其他内容，否则将受到惩罚。
        6. 如果用户输入的内容比较悲伤，只返回“depressed",不要有其他内容，否则将受到惩罚。
        7.如果用户输入的内容比较开心，只返回"cheerful",不要有其他内容，否则将受到惩罚。
        8. 只返回英文，不允许有换行符等其他内容，否则会受到惩罚。
        用户输入的内容是：{query}"""
        model=ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base=api_base_url,
            temperature=0,        
        )
        chain = ChatPromptTemplate.from_template(prompt) | model | StrOutputParser()
        result = chain.invoke({"query": query})
        self.emotion=result
        print("情绪分析结果:", result)
        return result

    def background_voice_synthesis(self,text:str,uid:str):
        pass

    async def generate_audio(self, text:str):
        import time
        # voice = "zh-CN-XiaoxiaoNeural"  # 选择语音，这里使用中文晓晓
        voice = "zh-CN-liaoning-XiaobeiNeural"
        output_file = f"audio_{int(time.time())}.mp3"
        voiceStyle=self.MOODS[self.emotion]["voiceStyle"]
        print(f'voiceStyle:{voiceStyle}')
        voice_params=self.EMOTION_PARAMS[voiceStyle]
        print(f'voice_params:{voice_params}')
        communicate = edge_tts.Communicate(text=text, 
                                           voice=voice, 
                                           rate=voice_params["rate"], 
                                           pitch=voice_params['pitch'], volume=voice_params['volume'])
        await communicate.save(output_file)
        with open(output_file, "rb") as f:
            audio_data = f.read()
        os.remove(output_file)  # 删除临时音频文件
        return audio_data

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query:str):
    master=Master()
    msg=master.run(query)
    unique_id=str(uuid.uuid4())
    return {"msg":msg, "unique_id":unique_id}
    # return {"response":"I am a chat"}

@app.post("/add_urls")
def add_urls(URL:str):
    try:
        normalized_url = validate_and_normalize_url(URL)
        loader = WebBaseLoader(normalized_url)
        docs = loader.load()
        cleaned_docs = [clean_document_content(doc) for doc in docs]
        print("docs:",cleaned_docs)
        
        if not cleaned_docs:
            raise ValueError(f"无法从URL加载内容: {normalized_url}")
        
        # 文本分割
        documents = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=50
        ).split_documents(cleaned_docs)
        print('documents:{documents}')
        # 创建嵌入模型（使用新的 HuggingFaceEmbeddings）
        def load_embeddings():
            """加载本地嵌入模型"""
            return HuggingFaceEmbeddings(
                model_name=MODEL_PATH,
                model_kwargs={"device": "cpu"},  # 使用CPU，如需GPU改为"cuda"
                encode_kwargs={"normalize_embeddings": True}
            )
        embeddings = load_embeddings()
        
        # 创建 Qdrant 向量数据库
        qdrant = Qdrant.from_documents(
            documents=documents,
            embedding=embeddings,
            path="./local_qdrant",
            collection_name="local_documents",
        )
        print("向量数据库创建完成")
        return {"ok": "添加成功！"}
    except Exception as e:
        print(f"加载文档失败: {e}")
        return {"failed": f"加载文档失败: {str(e)}"}
    

@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added!"}

@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 通信端点"""
    # 1. 接受客户端连接
    await websocket.accept()
    logger.info(f"新客户端连接,IP: {websocket.client.host}")

    try:
        while True:
            # 2. 接收用户消息（支持文本/JSON，这里示例用纯文本）
            user_message = await websocket.receive_text()
            logger.info(f"收到消息: {user_message}")

            # 3. 生成回复
            master=Master()
            msg=master.run(user_message)
            txt=msg["output"]
            # txt="我是聊天小助手，有需要叫我帮忙吗？"
            # 先发送文本响应
            await websocket.send_json({"text": txt})
            # 再发送音频二进制数据
            audio_data = await master.generate_audio(txt)
            await websocket.send_bytes(audio_data)
            # 4. 发送消息回复给客户端
            logger.info(f"已发送回复: {txt}")


    except WebSocketDisconnect:
        logger.info("客户端断开连接")
    except Exception as e:
        logger.error(f"通信异常: {str(e)}", exc_info=True)
    finally:
        # 清理资源（如关闭数据库连接等）
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",  # 允许外部访问（生产环境建议限制为内网IP）
        port=8000,
        # reload=True,      # 开发模式热重载（生产环境关闭）
        log_level="info"
    )
    # asyncio.run(get_voices())