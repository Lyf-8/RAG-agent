"""
阶段2-步骤2：父子块切分（Parent-Child Chunking）

职责：把结构化章节 → "父块 + 子块"两层结构：
- 子块（小）：只做检索定位，向量化后与用户问题匹配
- 父块（大）：子块命中后，把父块原文喂给 LLM 作为上下文

知识点讲解：
1. 为什么需要父子块？—— 解决"检索精度"和"上下文完整性"的矛盾
   - 块太大（阶段1的800字符）：向量是整块语义的平均，
     细节被稀释，匹配不准；但上下文完整
   - 块太小：向量聚焦、匹配准；但 LLM 拿到的上下文残缺，
     可能漏掉关键条件（比如审批规则只检索到"金额档位"，丢了"需附发票"）
   - 父子块 = 两全：小块负责"找准位置"，大块负责"给全信息"
   这是企业级 RAG 的标配切分策略（LangChain 叫 ParentDocumentRetriever）

2. 子块文本为什么要拼上标题路径？
   向量化的是 "标题路径 + 正文"，相当于告诉 Embedding 模型
   "这段话是在讲【考勤管理 > 请假审批】语境下的内容"，
   检索时问题"请假审批..."和它的语义距离会被显著拉近。
   这是零成本的召回增强手段。

3. 切块边界仍然用 RecursiveCharacterTextSplitter，
   只是换成两级：先切出父块（600），再把每个父块切成子块（200）。
"""
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    DATA_DIR,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """构造递归字符分割器（与阶段1相同的分隔符优先级策略）"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
    )


def build_parent_child_chunks(sections: list[dict], doc_name: str = "unknown") -> tuple[list[dict], list[dict]]:
    """
    结构化章节 → (父块列表, 子块列表)

    返回：
    parents = [{
        "parent_id": "manual_p0",
        "text": "父块原文（喂LLM用，不拼标题，保持干净）",
        "heading_path": "...",
        "doc_name": doc_name,
    }, ...]

    children = [{
        "child_id": "manual_p0_c0",
        "parent_id": "manual_p0",          # 关键：子块指向父块
        "text": "标题路径 + 子块正文（向量化/检索用）",
        "heading_path": "...",
        "doc_name": doc_name,
    }, ...]
    """
    parent_splitter = _make_splitter(PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)
    child_splitter = _make_splitter(CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)

    parents: list[dict] = []
    children: list[dict] = []

    for section in sections:
        # 1) 章节正文 → 父块（小章节通常 1 节 = 1 父块）
        parent_texts = parent_splitter.split_text(section["content"])

        for p_idx, p_text in enumerate(parent_texts):
            parent_id = f"{doc_name}_p{len(parents)}"
            parents.append({
                "parent_id": parent_id,
                "text": p_text,
                "heading_path": section["heading_path"],
                "doc_name": doc_name,
            })

            # 2) 父块 → 子块，子块文本拼上标题路径做"上下文增强"
            child_texts = child_splitter.split_text(p_text)
            for c_idx, c_text in enumerate(child_texts):
                # 拼标题路径：检索时语义带上章节归属
                enhanced = f"【{section['heading_path']}】\n{c_text}" if section["heading_path"] else c_text
                children.append({
                    "child_id": f"{parent_id}_c{c_idx}",
                    "parent_id": parent_id,
                    "text": enhanced,
                    "heading_path": section["heading_path"],
                    "doc_name": doc_name,
                })

    print(f"[ParentChildChunker] 父块 {len(parents)} 个，子块 {len(children)} 个"
          f"（平均每父块 {len(children) / max(len(parents), 1):.1f} 子块）")
    return parents, children


# ========== 运行测试 ==========
if __name__ == "__main__":
    from preprocess.parsing.structured_parser import parse_document_structured

    # 接步骤1：结构化解析 → 父子块切分
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    sections = parse_document_structured(doc_path)
    parents, children = build_parent_child_chunks(sections, doc_name="smart_office_manual")

    print("\n" + "=" * 70)
    print("父块示例（取第 5 个父块，对应'请假审批流程'章节）：")
    print("=" * 70)
    sample_p = parents[4]
    print(f"parent_id: {sample_p['parent_id']}")
    print(f"标题路径: {sample_p['heading_path']}")
    print(f"原文:\n{sample_p['text']}")

    print("\n" + "-" * 70)
    print("该父块下的子块（检索定位用，注意标题路径已拼入）：")
    print("-" * 70)
    for c in children:
        if c["parent_id"] == sample_p["parent_id"]:
            print(f"\nchild_id: {c['child_id']}  长度={len(c['text'])}")
            print(c["text"])

    print("\n" + "=" * 70)
    print(f"父块平均长度: {sum(len(p['text']) for p in parents) / len(parents):.0f} 字符")
    print(f"子块平均长度: {sum(len(c['text']) for c in children) / len(children):.0f} 字符")
    print("=" * 70)
