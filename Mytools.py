from langchain.agents import create_openai_tools_agent,AgentExecutor,tool
from langchain_openai import ChatOpenAI,OpenAI
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
#工具
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_core.output_parsers import JsonOutputParser
import requests
import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
import validators
from urllib.parse import urlparse, urljoin
from fastapi import HTTPException
import re
from langchain.docstore.document import Document

MODEL_PATH = "./models/all-MiniLM-L6-v2"
YUANFENJU_API_KEY = "your_token"
serp_api_key=os.getenv("SERPAPI_API_KEY")
deepseek_api_base="https://api.deepseek.com/v1"
deepseek_api_key=os.getenv("DEEPSEEK_API_KEY")

def clean_document_content(doc: Document) -> Document:
    """清理文档内容，去除多余的空白字符和特殊符号"""
    content = doc.page_content
    
    # 1. 合并连续的换行符为单个换行
    content = re.sub(r'\n{2,}', '\n', content)
    
    # 2. 合并连续的空格为单个空格
    content = re.sub(r'[ \t]+', ' ', content)
    
    # 3. 去除开头和结尾的空白字符
    content = content.strip()
    
    # 4. 保留段落结构（可选：如果需要更紧凑的文本，可以删除此步骤）
    content = re.sub(r'\n', ' ', content)  # 将所有换行符替换为空格
    
    # 5. 去除HTML实体（如果有残留）
    content = re.sub(r'&[a-z]+;', '', content)
    
    # 更新文档内容
    cleaned_doc = Document(
        page_content=content,
        metadata=doc.metadata
    )
    
    return cleaned_doc

@tool
def test():
    """Test tool"""
    return "test"

def validate_and_normalize_url(url: str) -> str:
    """验证并规范化URL"""
    # 添加协议头（如果缺失）
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
    
    # 验证URL格式
    if not validators.url(url):
        raise HTTPException(status_code=400, detail=f"无效的URL: {url}")
    
    # 处理国际化域名
    parsed = urlparse(url)
    if parsed.netloc:
        try:
            # 将域名转换为punycode格式
            punycode_netloc = parsed.netloc.encode('idna').decode('ascii')
            url = urljoin(f"{parsed.scheme}://{punycode_netloc}", parsed.path)
        except UnicodeError:
            raise HTTPException(status_code=400, detail=f"不支持的域名: {parsed.netloc}")
    
    return url


@tool
def search(query:str):
    """只有需要了解实时信息或不知道的事情的时候才会使用这个工具。"""
    # serp_api_key=os.getenv("SERPAPI_API_KEY")
    serp_api_key="your_token"
    print({"serp_api_key":serp_api_key})
    # serp = SerpAPIWrapper(serpapi_api_key=serp_api_key)
    # result=serp.run(query)
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "gl": "cn",  # 中国地区
        "hl": "zh-cn"  # 中文语言
    })
    headers = {
        'X-API-KEY': serp_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result=response.json()
    print("实时搜索结果:",result)
    return result


@tool
def get_info_from_local_db(query:str):
    """只有回答与2024年运势或者龙年运势相关的问题的时候,会使用这个工具"""
    # print("hello,wo来到了get_info_from_local_db工具了")
    def load_embeddings():
        """加载本地嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={"device": "cpu"},  # 使用CPU，如需GPU改为"cuda"
            encode_kwargs={"normalize_embeddings": True}
        )
    embeddings = load_embeddings()
    client = Qdrant(
        QdrantClient(path="./local_qdrant"),
        "local_documents",
        embeddings

    )
    retriever = client.as_retriever(search_type="mmr")
    result = retriever.get_relevant_documents(query)
    return result


@tool
def bazi_cesuan(query:str):
    """八字查询工具，只有在用户询问八字相关问题的时候才会使用这个工具"""
    url=f"https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"
    prompt = ChatPromptTemplate.from_template(
        """你是一个参数查询助手,根据用户输入内容找出相关的参数并按json格式返回。
        JSON字段如下: -"api_key":{YUANFENJU_API_KEY}, 
        -"name":"姓名", 
        -"sex":"性别,
        0表示男,1表示女,根据姓名判断", 
        -"type":"日历类型,0农历,1公里.默认1",
        -"year":"出生年份 
        例:1998", -"month":"出生月份 例 8", 
        -"day":"出生日期,例:8", 
        -"hours":"出生小时 例 14", 
        -"minute":"0",
        如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他的评论，用户输入:{query}"""
    )
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=deepseek_api_key,
        openai_api_base=deepseek_api_base,
        temperature=0
        ) | parser
    data = chain.invoke({"query":query})
    print("八字查询结果:",data)
    result = requests.post(url,data={"data":data, "api_key":YUANFENJU_API_KEY})
    if result.status_code == 200:
        print("=====返回数据=====")
        print(result.json())
        try:
            json=result.json()
            rstr="八字为:"+json["data"]["bazi_info"]["bazi"]
            return rstr
        except Exception as e:
            return "八字查询失败，可以时你忘记询问用户姓名或者出生年月日时了"
            
    else:
        return "技术错误，请告诉用户"
    
@tool
def yaoyigua():
    """只有用户想要占卜抽签的时候才会使用这个工具。"""
    api_key = YUANFENJU_API_KEY
    url = f"https://api.yuanfenju.com/index.php/v1/Zhanbu/yaogua"
    result = requests.post(url,data={"api_key":api_key})
    if result.status_code == 200:
        print("====返回数据=====")
        print(result.json())
        returnstring = json.loads(result.text)
        
        # common_desc1=returnstring["data"]["common_desc1"]
        # common_desc2 = returnstring["data"]["common_desc2"]
        # common_desc3 = returnstring["data"]["common_desc3"]
    
        image = returnstring["data"]["image"]
        print('returnstrnig:',returnstring)
        print("卦图片:",image)
        return  returnstring
    else:
        return "技术错误，请告诉用户稍后再试。"

@tool
def jiemeng(query:str):

    """只有用户想要解梦的时候才会使用这个工具,需要输入用户梦境的内容，如果缺少用户梦境的内容则不可用。"""
    api_key = YUANFENJU_API_KEY
    url =f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    LLM = OpenAI(model_name="deepseek-chat",
                 openai_api_key=deepseek_api_key,
                 openai_api_base=deepseek_api_base,
                 temperature=0)
    prompt = PromptTemplate.from_template("根据内容提取1个关键词，只返回关键词，内容为:{topic}")
    prompt_value = prompt.invoke({"topic":query})
    keyword = LLM.invoke(prompt_value)
    print("提取的关键词:",keyword)
    result = requests.post(url,data={"api_key":api_key,"title_zhougong":keyword})
    if result.status_code == 200:
        print("====返回数据=====")
        print(result.json())
        returnstring = json.loads(result.text)
        return returnstring
    else:
        return "技术错误，请告诉用户稍后再试。"

@tool
def tongziming(query:str):
    """童子命查询工具,只有用户想询问童子命相关问题时，调用此工具"""
    url=f"https://api.yuanfenju.com/index.php/v1/Gongju/tongzi"
    prompt = ChatPromptTemplate.from_template(
        """你是一个参数查询助手,根据用户输入内容找出相关的参数并按json格式返回。
        JSON字段如下: 
        -"api_key":{YUANFENJU_API_KEY}, 
        -"name":"姓名", 
        -"sex":"性别, 0表示男,1表示女,根据姓名判断", 
        -"type":"日历类型,0农历,1公里.默认1",
        -"year":"出生年份 例:1998", 
        -"month":"出生月份 例:8", 
        -"day":"出生日期 例:8", 
        -"hours":"出生小时 例:14", 
        -"minute":"0",
        如果没有找到相关参数，则需要提醒用户告诉你这些内容，
        只返回数据结构，不要有其他的评论，用户输入:{query}"""
    )
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=deepseek_api_key,
        openai_api_base=deepseek_api_base,
        temperature=0
        ) | parser
    data = chain.invoke({"query":query})
    print("八字查询结果:",data)
    result = requests.post(url,data={"data":data})
    if result.status_code == 200:
        print("=====返回数据=====")
        print(result.json())
        try:
            json=result.json()
            rstr="童子命："+json["data"]["tongziming"]["description"]
            print(rstr)
            return rstr
        except Exception as e:
            return "童子命查询失败"
    else:
        return "技术错误，请告诉用户"

if __name__=="__main__":
    print("hello")