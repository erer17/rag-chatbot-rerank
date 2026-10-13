from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import requests
import os
from fastembed import TextEmbedding

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("company_docs")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# 默认模型名
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# 本地向量模型
embed_model = TextEmbedding("BAAI/bge-small-zh-v1.5")

def get_embedding(text):
    return list(embed_model.embed([text]))[0].tolist()

# 本地 rerank（用到才延迟加载）
_reranker = None
def rerank(query, documents, top_n=3):
    global _reranker
    try:
        if _reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            # 如果 Render 512MB 内存不够，加载模型可能报错
            _reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        
        scores = list(_reranker.rerank(query, documents))
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]
    except Exception as e:
        print(f"Rerank 失败，回退到原始文档顺序: {e}")
        # 如果 Rerank 模型加载或计算失败，降级退回前 top_n 个文档，保证接口不崩
        return documents[:top_n]

def ask_gemini(prompt):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 GEMINI_API_KEY 环境变量")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    
    try:
        # ⬅️ 修改点 1：设置连接超时 10 秒，读取响应超时 120 秒，避免 60s 直接抛出 ReadTimeout
        resp = requests.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=(10, 120) 
        )
        data = resp.json()

        # ⬅️ 修改点 2：更严谨的解析逻辑
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            # 打印详细错误日志到 Render 控制台
            print(f"Gemini API Error Response: {data}")
            error_msg = data.get("error", {}).get("message", str(data))
            raise HTTPException(status_code=500, detail=f"Gemini 生成失败: {error_msg}")

    except requests.exceptions.Timeout:
        # ⬅️ 修改点 3：捕获超时，优雅抛出 HTTP 504 错误
        raise HTTPException(status_code=504, detail="Gemini API 响应超时，请重试。")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"网络请求失败: {str(e)}")


class Question(BaseModel):
    question: str
    use_rerank: bool = True  # 默认关闭


@app.post("/ask")
def ask(q: Question):
    try:
        # 1. 向量检索
        query_embedding = get_embedding(q.question)
        results = collection.query(query_embeddings=[query_embedding], n_results=10)
        
        candidates = results.get("documents", [[]])[0]
        if not candidates:
            return {"answer": "知识库中未查找到相关紫微斗数资料。", "used_rerank": q.use_rerank, "retrieved_docs": []}

        # 2. 是否使用 Rerank
        top_docs = rerank(q.question, candidates, top_n=3) if q.use_rerank else candidates[:3]
        context = "\n".join(top_docs)

        # 3. 构造 Prompt 并调用 AI
        prompt = (
            f"根据以下紫微斗数文件回答问题，若文件没提到就说不知道。\n"
            f"文件内容:\n{context}\n\n"
            f"问题:{q.question}\n"
            f"回答:"
        )
        
        answer = ask_gemini(prompt)
        return {"answer": answer, "used_rerank": q.use_rerank, "retrieved_docs": top_docs}

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        print(f"全局未捕获异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")