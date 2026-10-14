# query.py - 提问：从知识库检索最相关的片段
# 用法：python query.py（进入交互，可连续提问，输入 exit 退出）
import os       #操作系统工具

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys      #系统参数工具

from sentence_transformers import SentenceTransformer   #转向量的模型

import chromadb #引入向量数据库

import config   #自己的配置文件


def search(question, top_k=config.TOP_K):   #定义查找函数
    embedder = SentenceTransformer(config.EMBED_MODEL)  #加载向量化模型
    client = chromadb.PersistentClient(path=str(config.DB_DIR)) #打开本地知识库
    collection = client.get_collection("paper_kb")  #打开知识集合
    q_emb = embedder.encode([question]) #把问题变为向量
    return collection.query(query_embeddings=q_emb, n_results=top_k)    #返回最接近片段


def ask_llm(question, context):
    """可选：把检索片段交给大模型，整理成一段中文回答。"""
    import requests     #引入工具，用于发网络请求

    prompt = (
        "请仅依据下面给出的英文资料，用中文简洁回答我的问题，并标注资料出处。\n\n"
        f"资料：\n{context}\n\n问题：{question}"
    )
    resp = requests.post(       #向大模型发送网络请求
        f"{config.LLM_API_BASE}/chat/completions",      #地址
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
        json={                  #请求体
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,     #60s内回应
    )
    resp.raise_for_status()     #出错则报错
    return resp.json()["choices"][0]["message"]["content"]      #返回中文回答


def answer(question):       #拼接所有内容
    res = search(question)      #取出问题
    docs = res["documents"][0]  #取出片段正文
    metas = res["metadatas"][0] #取出片段位置
    context = "\n\n".join(
        f"[{m['source']} 第{m['chunk_id']}块]\n{d}" for d, m in zip(docs, metas)
    )

    if config.LLM_API_BASE and config.LLM_API_KEY:
        print("=== 中文回答 ===")
        print(ask_llm(question, context))
        print("\n=== 依据片段 ===")
    else:
        print("=== 检索到的相关片段（英文原文依据） ===")       #有key输出中文，没有直接输出英文
    print(context)      #输出正文片段


def main():
    if len(sys.argv) >= 2:
        answer(" ".join(sys.argv[1:]))
        return

    print("==== 论文知识库（中文提问；输入 exit 退出） ====")
    while True:
        try:
            q = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出", "q"):
            break
        print()
        answer(q)


if __name__ == "__main__":
    main()
