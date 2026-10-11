from huggingface_hub import InferenceClient
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os

# HF_TOKEN = os.environ.get("HF_TOKEN")
# hf_client = InferenceClient(token=HF_TOKEN)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # 模型名可用环境变量改

#def get_embedding(text):
#    result = hf_client.feature_extraction(text, model="BAAI/bge-small-zh-v1.5")
#    if hasattr(result, 'ndim') and result.ndim == 2:
#        result = result.mean(axis=0)
#    return result.tolist()

from fastembed import TextEmbedding
embed_model = TextEmbedding("BAAI/bge-small-zh-v1.5")

def get_embedding(text):
    return list(embed_model.embed([text]))[0].tolist()

pdf_folder = "."
all_documents = []
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        reader = PdfReader(os.path.join(pdf_folder, filename))
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        chunks = splitter.split_text(full_text)
        all_documents.extend(chunks)

documents = all_documents
print(f"切成了 {len(documents)} 个段落，开始用API算向量...")

client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("company_docs")  # 清掉旧的，避免混杂
except:
    pass
collection = client.get_or_create_collection("company_docs")

embeddings = []
for i, doc in enumerate(documents):
    embeddings.append(get_embedding(doc))
    if i % 10 == 0:
        print(f"已处理 {i}/{len(documents)} 段")

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc{i}" for i in range(len(documents))]
)

print("向量数据库建立完成！")

# from sentence_transformers import SentenceTransformer
# import chromadb
# from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# import os

# pdf_folder = "."  # 把所有PDF放这个文件夹
# all_documents = []

# splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
# for filename in os.listdir(pdf_folder):
#     if filename.endswith(".pdf"):
#         reader = PdfReader(os.path.join(pdf_folder, filename))
#         full_text = ""
#         for page in reader.pages:
#             full_text += page.extract_text() + "\n"
#         chunks = splitter.split_text(full_text)
#         all_documents.extend(chunks)

# documents = all_documents

# ① 读取PDF全部文字
#reader = PdfReader("company_policy.pdf")  # 换成你的PDF文件名，放在同一个文件夹
#full_text = ""
#for page in reader.pages:
#    full_text += page.extract_text() + "\n"

#print(f"共读取 {len(full_text)} 个字")

# ② 把长文字切成小段落（每段约600字，段落间重叠80字避免断句丢失上下文）
#splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
#documents = splitter.split_text(full_text)

# print(f"切成了 {len(documents)} 个段落")

# ③ 转成向量，存进数据库（跟之前一样）
# model = SentenceTransformer('all-MiniLM-L6-v2')
# client = chromadb.PersistentClient(path="./chroma_db")
# collection = client.get_or_create_collection("company_docs")

# embeddings = model.encode(documents).tolist()
# collection.add(
#     documents=documents,
#     embeddings=embeddings,
#     ids=[f"doc{i}" for i in range(len(documents))]
# )

# print("向量数据库建立完成！")