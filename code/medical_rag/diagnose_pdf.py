"""
诊断 Unstructured PDF 提取问题
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# 禁用 Unstructured 的复杂 PDF 检测，强制使用 fast 策略
os.environ["UNSTRUCTURED_IO_THRESHOLD"] = "0"

from unstructured.partition.pdf import partition_pdf

pdf_path = "../../data/medical_rag/internal/内科学(第10版).pdf"

print("=" * 60)
print("诊断 1: 检查 PDF 文件")
print("=" * 60)
print(f"文件存在: {Path(pdf_path).exists()}")
print(f"文件大小: {Path(pdf_path).stat().st_size / 1024 / 1024:.1f} MB")

# 用 pypdf 检查页数和基本信息
from pypdf import PdfReader
reader = PdfReader(pdf_path)
print(f"PDF 页数: {len(reader.pages)}")

# 检查前3页是否有文字
for i in range(min(3, len(reader.pages))):
    text = reader.pages[i].extract_text()
    print(f"  第{i+1}页文字长度: {len(text) if text else 0}")
    if text:
        print(f"  前100字: {text[:100]}")

print("\n" + "=" * 60)
print("诊断 2: Unstructured fast 策略")
print("=" * 60)

try:
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",
        languages=["chi_sim", "eng"],
    )
    print(f"元素数量: {len(elements)}")
    for i, elem in enumerate(elements[:5]):
        print(f"  [{i+1}] category={elem.category}, text={str(elem)[:80]}")
except Exception as e:
    print(f"fast 策略失败: {e}")

print("\n" + "=" * 60)
print("诊断 3: Unstructured ocr_only 策略")
print("=" * 60)

try:
    elements = partition_pdf(
        filename=pdf_path,
        strategy="ocr_only",
        languages=["chi_sim", "eng"],
    )
    print(f"元素数量: {len(elements)}")
    for i, elem in enumerate(elements[:5]):
        print(f"  [{i+1}] category={elem.category}, text={str(elem)[:80]}")
except Exception as e:
    print(f"ocr_only 策略失败: {e}")
