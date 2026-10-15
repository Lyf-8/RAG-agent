"""
步骤2：基础字符切分

职责：将长文本 → 固定大小的小块（chunks），供 Embedding 向量化

核心工具：LangChain 的 RecursiveCharacterTextSplitter（递归字符分割器）
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def get_chunker(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """
    创建一个递归字符分割器实例

    参数说明：
    - chunk_size: 每块最大字符数（默认800，来自 settings.py）
    - chunk_overlap: 相邻块重叠字符数（默认80，10%重叠率）

    分隔符优先级（从高到低）：
    1. "\\n\\n" — 段落边界（最优先，保持段落完整）
    2. "\\n"    — 行边界
    3. " "      — 空格边界
    4. ""       — 字符级（最后手段，硬切）
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,  # 保留分隔符（换行符），避免拼接后丢失格式
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    将纯文本切分为 chunks 列表

    输入：parse_document() 返回的纯文本
    输出：["块1内容", "块2内容", "块3内容", ...]

    使用方式：
        text = parse_document("manual.md")
        chunks = chunk_text(text)
        print(f"切分为 {len(chunks)} 块")
    """
    splitter = get_chunker(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


# ========== 运行测试 ==========
if __name__ == "__main__":
    # 接步骤1：解析 → 切分
    from preprocess.parsing.parser import parse_document
    from config.settings import DATA_DIR

    # 1. 解析文档
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    text = parse_document(doc_path)

    # 2. 切分
    chunks = chunk_text(text)

    print("\n" + "=" * 60)
    print(f"切分结果：共 {len(chunks)} 块")
    print(f"参数：chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print("=" * 60)

    # 展示每块的编号和前80字符
    for i, chunk in enumerate(chunks):
        print(f"\n[块 {i + 1}] 长度={len(chunk)} 字符")
        print(f"  预览: {chunk[:80]}...")

    print("\n" + "=" * 60)
    print(f"平均块长: {sum(len(c) for c in chunks) / len(chunks):.0f} 字符")
