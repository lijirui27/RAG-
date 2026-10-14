# ingest.py - 建库：把 papers 里的 PDF 解析、清洗、切块、向量化后写入本地向量库
# 用法：python ingest.py
import os

# 用国内镜像下载模型，避免直连 huggingface.co 超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
from pathlib import Path

import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb

import config

# ---------- OCR 组件（处理扫描件/图片型 PDF） ----------
try:
    import pymupdf  # 用 pymupdf
    from rapidocr_onnxruntime import RapidOCR
    _OCR = RapidOCR()  # 中英文识别引擎（模型内置，无需联网）
except Exception:
    _OCR = None


# ---------- 清洗层：修复 OCR 常见问题，提升检索质量 ----------
_PAGE_HEADER = re.compile(
    r"(european journal|elsevier|journal of operational|primary operations|"
    r"the hot strip mill production scheduling|blast furnace|melt shop)", re.I
)
# 词连写修复：小写/数字后紧跟大写字母（OCR 常把空格吃掉，如 "Departmentof"）
_WORD_JOIN = re.compile(r"([a-zà-ÿ0-9])([A-ZÀ-Ý])")
# 多余空格合并
_MULTI_SPACE = re.compile(r"[ \t]+")
# 纯页码 / 卷期行（如 "318"、"317-335"）
_PAGE_NUM = re.compile(r"\d{1,4}(?:-\d{1,4})?")


def clean_text(text):
    """清洗一段 OCR 文本：去页眉页码、拆"小写+大写"连写、过滤噪声行。"""
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        # 去掉纯页码/卷期行
        if _PAGE_NUM.fullmatch(s):
            continue
        # 去掉期刊页眉等重复性标题行（保留正文）
        if _PAGE_HEADER.search(s) and len(s) < 90:
            continue
        # 修复"小写+大写"词连写（如 Departmentof -> Department of）
        s = _WORD_JOIN.sub(r"\1 \2", s)
        # 合并多余空格
        s = _MULTI_SPACE.sub(" ", s)
        # 过滤过短噪声（单个字符、纯标点）
        if len(s) <= 1:
            continue
        lines.append(s)
    return "\n".join(lines)


def ocr_pdf(pdf_path, zoom=2.0):
    """把图片型 PDF 逐页渲染成图，用 OCR 识别成文字。"""
    import numpy as np
    doc = pymupdf.open(pdf_path)
    texts = []
    for page in doc:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        result, _ = _OCR(img)
        texts.append("\n".join(r[1] for r in result) if result else "")
    return texts


def extract_text(pdf_path):
    """提取 PDF 全文，返回 [(页码, 清洗后的文本), ...]。文字型直接读，图片型走 OCR。"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append((num, text))

    if all(not t.strip() for _, t in pages):  # 所有页都读不到文字 -> 扫描件
        if _OCR is None:
            raise RuntimeError("扫描件需要 OCR，请先安装 rapidocr_onnxruntime pymupdf")
        print("  检测到扫描件，正在用 OCR 识别文字（较慢，请耐心等待）...")
        ocr_texts = ocr_pdf(pdf_path)
        pages = [(i + 1, t) for i, t in enumerate(ocr_texts)]

    pages = [(num, clean_text(t)) for num, t in pages]
    return pages


def split_into_chunks(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """把长文本切成带重叠的片段，每片自带上下文。"""
    text = re.sub(r"\n{2,}", "\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build():
    pdf_files = sorted(config.PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"错误：{config.PAPERS_DIR} 里没有 PDF 文件，请先把论文放进去")
        return

    print("加载嵌入模型，首次会自动下载（走国内镜像）...")
    embedder = SentenceTransformer(config.EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    # 重建：先删掉旧 collection，避免旧脏数据混入新库
    try:
        client.delete_collection("paper_kb")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        "paper_kb", metadata={"hnsw:space": "cosine"}
    )

    all_texts, all_metas = [], []
    for pdf in pdf_files:
        print(f"处理 {pdf.name} ...")
        pages = extract_text(pdf)
        full = "\n".join(t for _, t in pages)
        if len(full.strip()) < 50:
            print(f"  警告：{pdf.name} 未提取到文本")
            continue
        for idx, chunk in enumerate(split_into_chunks(full)):
            all_texts.append(chunk)
            all_metas.append({"source": pdf.name, "chunk_id": idx})

    if not all_texts:
        print("没有可入库的内容，请检查 PDF。")
        return

    print(f"共 {len(all_texts)} 个片段，开始向量化入库...")
    embeddings = embedder.encode(all_texts, batch_size=16, show_progress_bar=True)
    ids = [f"c{i}" for i in range(len(all_texts))]
    collection.add(ids=ids, documents=all_texts, embeddings=embeddings, metadatas=all_metas)
    print(f"入库完成，共 {collection.count()} 条知识片段。")


if __name__ == "__main__":
    build()
