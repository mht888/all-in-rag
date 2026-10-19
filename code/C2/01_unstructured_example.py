from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
# elements = partition(
#     filename=pdf_path,
#     content_type="application/pdf"
# )

elements = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only",
    languages=["chi_sim", "eng"]
)
##`hi_res` 版面分析 视觉模型 识别标题/表格/图片等元素，扫描页再叠加 OCR；可识别表格结构、图片 最慢 复杂排版、扫描件、需要表格结构 
# `ocr_only` 不使用版面模型 ，把每一页渲染成图片后整页跑 Tesseract OCR 提取文字 慢 纯扫描件，且不需要区分元素类型

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)