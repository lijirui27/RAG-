# config.py - 配置，需要调参数就改这里
from pathlib import Path

# 项目根目录（本文件所在目录）
BASE_DIR = Path(__file__).parent

# 放论文 PDF 的文件夹
PAPERS_DIR = BASE_DIR / "papers"

# 向量数据库持久化目录（运行 ingest.py 后自动生成）
DB_DIR = BASE_DIR / "db" 

# 嵌入模型：多语言模型，中文提问也能匹配英文内容（解决跨语言检索）
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 切块参数
CHUNK_SIZE = 500       # 每个知识块约多少字符
CHUNK_OVERLAP = 80     # 相邻块重叠，避免切断语义

# 检索时最多返回几个片段（调大一点，藏在角落的答案更容易被捞到）
TOP_K = 6

# ---------- DeepSeek：生成中文连贯回答 ----------
# 配置好下面三项后，提问会先检索、再由 DeepSeek 整理成一句中文回答。
LLM_API_BASE = "https://api.deepseek.com/v1"  # DeepSeek OpenAI 兼容接口
LLM_API_KEY = "sk-0585693303464bc393dab2386d25d2af"  # DeepSeek Key
LLM_MODEL = "deepseek-chat"
