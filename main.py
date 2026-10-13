from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import requests
import os
from fastembed import TextEmbedding   # ⬅️ 取代 huggingface_hub 的 InferenceClient

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("company_docs")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")  # 模型名可用环境变量改

# ⬇️ 本地算向量,跟 build_db.py 用同一个模型
embed_model = TextEmbedding("BAAI/bge-small-zh-v1.5")

def get_embedding(text):
    return list(embed_model.embed([text]))[0].tolist()

# ⬇️ 本地rerank(第一次用到才加载,避免平时占内存)
_reranker = None
def rerank(query, documents, top_n=3):
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
    scores = list(_reranker.rerank(query, documents))
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]

def ask_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    resp = requests.post(
        url,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return f"生成失败: {data}"   # 额度/模型名错误会在这里显示


class Question(BaseModel):
    question: str
    use_rerank: bool = False   # 默认关闭,本地测试对比时才设成 true

@app.post("/ask")
def ask(q: Question):
    query_embedding = get_embedding(q.question)
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    candidates = results["documents"][0]

    top_docs = rerank(q.question, candidates, top_n=3) if q.use_rerank else candidates[:3]
    context = "\n".join(top_docs)

    prompt = (f"根据以下紫微斗数文件回答问题,若文件没提到就说不知道。\n"
              f"文件内容:{context}\n问题:{q.question}\n回答:")
    answer = ask_gemini(prompt)
    return {"answer": answer, "used_rerank": q.use_rerank, "retrieved_docs": top_docs}
