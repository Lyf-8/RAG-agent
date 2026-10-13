"""
阶段2-步骤1：标题层级识别结构化解析

职责：将 Markdown 文档 → 带"标题路径"元数据的结构化章节列表，
     替代阶段1的"整篇纯文本"，为父子块切分提供结构基础。

知识点讲解：
1. 阶段1的痛点：整篇文档清洗成一大段纯文本后切分，
   切出来的块是"无身份的"——不知道它属于哪一章哪一节。
   比如"直属主管审批即可生效"这句话，脱离了"2.2 请假审批流程"这个标题，
   向量检索时就丢掉了最重要的上下文线索。

2. 什么是"标题路径"（Heading Path）？
   把每个章节内容挂上它的层级归属，例如：
   "智能办公系统用户手册 > 2. 考勤管理模块 > 2.2 请假审批流程"
   这个路径有两个用途：
   - 检索增强：把标题路径拼进块的文本里一起向量化，
     相当于给每块内容"补上章节上下文"，召回更准
   - 溯源展示：答案引用时能显示"出自第几章"，企业场景必须可审计

3. 实现思路（状态机扫描）：
   逐行扫描 Markdown，遇到 # / ## / ### 就更新"当前标题栈"，
   普通文本行归入当前最新标题下，自然形成"章节 → 内容"的归属关系。
   这是编译原理里最简单的单遍扫描思想，不依赖任何 NLP 模型。
"""
import re
from pathlib import Path

from config.settings import DATA_DIR

# 匹配 Markdown 标题行：# / ## / ###（支持到6级，够用）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_markdown_structured(file_path: str | Path) -> list[dict]:
    """
    解析 Markdown → 结构化章节列表

    返回格式：
    [
        {
            "content": "章节正文（不含标题行本身）",
            "heading_path": "手册 > 2. 考勤管理模块 > 2.2 请假审批流程",
            "h1": "智能办公系统用户手册",   # 无该级标题则为 ""
            "h2": "2. 考勤管理模块",
            "h3": "2.2 请假审批流程",
            "level": 3,                    # 本章节最深标题级别
        },
        ...
    ]

    设计要点：
    - 用"标题栈"维护当前层级：遇到 H2 就替换栈中 H2 及以下，天然处理层级跳转
    - 前言（第一个标题之前的内容）单独成节，heading_path 为空字符串
    """
    file_path = Path(file_path)
    raw = file_path.read_text(encoding="utf-8")
    lines = raw.split("\n")

    # 标题栈：stack[level-1] = 标题文本；如 stack[0]=H1, stack[1]=H2
    heading_stack: list[str] = []
    sections: list[dict] = []
    # 当前章节累积的正文行
    current_lines: list[str] = []

    def flush_section():
        """把当前累积的正文落盘为一个章节（内部函数，避免重复代码）"""
        # 去除首尾空行
        content = "\n".join(current_lines).strip()
        if not content:
            return
        # 标题路径：用 " > " 连接，空级跳过
        path_parts = [h for h in heading_stack if h]
        sections.append({
            "content": content,
            "heading_path": " > ".join(path_parts),
            "h1": heading_stack[0] if len(heading_stack) > 0 else "",
            "h2": heading_stack[1] if len(heading_stack) > 1 else "",
            "h3": heading_stack[2] if len(heading_stack) > 2 else "",
            "level": len(heading_stack),
        })

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            # 遇到新标题：先把上一章节正文落盘
            flush_section()
            current_lines = []

            level = len(m.group(1))   # "#" 的个数 = 层级（1=H1）
            title = m.group(2)
            # 更新标题栈：当前级替换，更深层级全部弹出
            # 例：栈为 [考勤, 打卡方式]，遇到新 H2 → 栈变为 [考勤, 新标题]，H3 被弹出
            heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level:
                heading_stack.append("")
            heading_stack[level - 1] = title
        else:
            current_lines.append(line)

    # 循环结束，最后一节落盘
    flush_section()

    print(f"[StructuredParser] {file_path.name}: 解析出 {len(sections)} 个结构化章节")
    return sections


# 路由表（与阶段1 parser.py 风格一致，方便扩展 PDF/DOCX 的结构化解析）
STRUCTURED_PARSERS = {
    ".md": parse_markdown_structured,
}


def parse_document_structured(file_path: str | Path) -> list[dict]:
    """统一入口：按后缀选择结构化解析器"""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    parser = STRUCTURED_PARSERS.get(suffix)
    if parser is None:
        raise ValueError(
            f"结构化解析暂不支持: {suffix}，当前支持: {list(STRUCTURED_PARSERS.keys())}"
        )
    return parser(file_path)


# ========== 运行测试 ==========
if __name__ == "__main__":
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    sections = parse_document_structured(doc_path)

    print("\n" + "=" * 70)
    print(f"共解析出 {len(sections)} 个章节，逐节预览：")
    print("=" * 70)
    for i, sec in enumerate(sections, 1):
        print(f"\n[章节 {i}] 层级=H{sec['level']}  正文长度={len(sec['content'])} 字符")
        print(f"  标题路径: {sec['heading_path'] or '(前言，无标题)'}")
        preview = sec["content"].replace("\n", " ")[:60]
        print(f"  正文预览: {preview}...")

    # 统计：各层级章节数量
    print("\n" + "-" * 70)
    level_count = {}
    for sec in sections:
        level_count[sec["level"]] = level_count.get(sec["level"], 0) + 1
    for lv in sorted(level_count):
        print(f"  H{lv} 章节数: {level_count[lv]}")
    print("=" * 70)
