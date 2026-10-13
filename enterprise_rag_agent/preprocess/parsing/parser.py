"""
步骤1：基础文档解析与清洗

职责：将 PDF/Markdown 原始文档 → 干净的纯文本
- 去除页眉页脚、多余换行、脏数据
- 保留纯文本内容，供后续切分使用

知识点讲解：
1. 为什么需要"清洗"？原始文档中常有页码、页眉、乱码、连续空行等噪声，
   如果不清洗直接切分，会导致向量质量下降，检索时召回无关内容。
2. Markdown 解析用 LangChain 的 UnstructuredMarkdownLoader，
   PDF 解析用 pypdf，两者最终都输出纯文本。
3. POC 阶段只做最简清洗：去多余空行 + strip，阶段2会升级为标题层级识别。
"""
import re
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader

# 统一配置入口
from config.settings import DATA_DIR


def clean_text(text: str) -> str:
    """
    基础文本清洗 —— POC 级别，只做三件事：
    1. 去除连续空行（3个以上换行 → 2个）
    2. 去除每行首尾空白
    3. 整体 strip

    为什么只做这三件事？
    POC 阶段追求快速跑通，不引入复杂正则，避免误删有效内容。
    阶段2会升级为标题识别 + 结构化解析。
    """
    # 1. 连续空行压缩：3个以上换行 → 2个
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 2. 每行 strip
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # 3. 整体 strip
    text = text.strip()
    return text


def parse_markdown(file_path: str | Path) -> str:
    """
    解析 Markdown 文档 → 纯文本

    POC 阶段直接用 TextLoader 读取（Markdown 本质是带格式的纯文本），
    保留 Markdown 标题标记（# ## ###），阶段2会用它做标题层级识别。
    """
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    raw_text = docs[0].page_content if docs else ""
    return clean_text(raw_text)


def parse_pdf(file_path: str | Path) -> str:
    """
    解析 PDF 文档 → 纯文本

    使用 pypdf 逐页提取文本，拼接后清洗。
    POC 阶段不处理表格、图片、扫描件（OCR），仅提取纯文本层。
    """
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    # 多页 PDF 会返回多个 Document，按页拼接
    raw_text = "\n\n".join(doc.page_content for doc in docs)
    return clean_text(raw_text)


def parse_text(file_path: str | Path) -> str:
    """
    解析纯文本文件 → 直接读取 + 清洗
    """
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    raw_text = docs[0].page_content if docs else ""
    return clean_text(raw_text)


# 文件后缀 → 解析函数 的路由表（方便扩展）
PARSERS = {
    ".md": parse_markdown,
    ".pdf": parse_pdf,
    ".txt": parse_text,
}


def parse_document(file_path: str | Path) -> str:
    """
    统一入口：根据文件后缀自动选择解析器

    使用方式：
        text = parse_document("data/raw_docs/manual.md")
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(
            f"不支持的文件格式: {suffix}，当前支持: {list(PARSERS.keys())}"
        )

    print(f"[Parser] 正在解析: {file_path.name}")
    text = parser(file_path)
    print(f"[Parser] 解析完成，纯文本长度: {len(text)} 字符")
    return text


# ========== 运行测试 ==========
if __name__ == "__main__":
    # 用测试文档跑一遍
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    text = parse_document(doc_path)

    # 预览前500字符
    print("\n" + "=" * 60)
    print("解析结果预览（前500字符）：")
    print("=" * 60)
    print(text[:500])
    print("=" * 60)
    print(f"\n总长度: {len(text)} 字符")
