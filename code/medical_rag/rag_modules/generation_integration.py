"""
医学RAG系统 - 生成集成模块
负责：LLM集成、LLM查询路由（多维度结构化分析）、查询重写、多模板回答生成（支持流式输出）
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Generator

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """
    生成集成模块 - 负责LLM集成、查询路由和回答生成

    整体流程（由 main.py 的 ask_question 驱动）：
      1. query_router(query)           → LLM 分析路由类型/紧急程度/专业级别/医学实体
      2. query_rewrite(query)          → LLM 判断是否需要重写查询（紧急情况跳过）
      3. generate_answer(...)          → 按路由类型选择模板，生成回答（支持流式）
         ├─ emergency  → 急救模板（最高优先级，前置紧急提醒）
         ├─ diagnosis  → 症状分析模板
         ├─ treatment  → 治疗方案模板
         ├─ prevention → 预防保健模板
         └─ general    → 通用科普模板
      4. _build_context(docs)          → 将检索到的文档格式化为上下文（限长 6000 字符）

    关键设计：
      - 路由分析由 LLM 完成，返回结构化 JSON（route/urgency/expertise/entities 四维度）
      - 专业级别（expertise）控制回答语气：patient 用通俗语言，professional 用医学术语
      - 流式输出使用 invoke + 逐段 yield，避免 SSE 流卡死问题（与 C8 保持一致）
      - LLM 返回的 JSON 可能包含 markdown 围栏，需额外剥离处理
    """

    # 路由字段的有效取值
    VALID_ROUTES = ('emergency', 'diagnosis', 'treatment', 'prevention', 'general')
    VALID_URGENCY = ('critical', 'high', 'normal')
    VALID_EXPERTISE = ('patient', 'professional')

    # 不同专业级别的表达要求
    EXPERTISE_INSTRUCTIONS = {
        'patient': "用户是普通患者，请使用通俗易懂的语言，专业术语出现时用括号简要解释，语气平和耐心。",
        'professional': "用户是医学专业人员，请使用规范医学术语，表述简明准确。",
    }

    # 各路由类型的回答模板
    TEMPLATES = {
        'emergency': """
你是一位专业的医学急救指导助手。用户描述的情况可能属于紧急状况。

用户问题: {question}

相关医学参考资料:
{context}

表达要求: {expertise_instruction}

请立即按以下结构给出指导：

## 🚨 紧急情况提醒

**立即行动：**
1. 拨打 120 急救电话（或请身边人立即拨打）
2. [根据具体情况给出关键急救措施]

**等待救援时：**
[简要列出保持生命体征的注意事项]

⚠️ 以上为急救常识指导，请以急救人员和医生的指导为准。

回答:""",

        'diagnosis': """
你是一位专业的医学知识助手。请根据医学参考资料，分析用户描述的症状。

用户问题: {question}

相关医学参考资料:
{context}

表达要求: {expertise_instruction}

请按以下结构回答：

## 🔍 症状分析
[基于参考资料，分析可能的原因，列出可能性较大的情况]

## 📋 典型表现
[相关疾病的常见症状描述]

## 🩺 建议检查
[推荐的相关检查项目]

## 🏥 就医建议
[建议就诊科室，以及出现哪些情况需立即就医]

⚠️ 重要提醒：以上分析仅供参考，不能替代专业医疗诊断，请及时就医并由医生做出判断。

回答:""",

        'treatment': """
你是一位专业的医学知识助手。请根据医学参考资料，提供治疗相关的信息。

用户问题: {question}

相关医学参考资料:
{context}

表达要求: {expertise_instruction}

请按以下结构回答：

## 💊 治疗原则
[基于参考资料概述治疗思路]

## 📖 常用治疗手段
[常用药物/手术/其他治疗方式，如涉及药物请注明"须遵医嘱使用"]

## ⚠️ 注意事项
[禁忌、不良反应、特殊人群提示]

## 🌱 生活配合
[饮食、运动等辅助治疗的日常建议]

⚠️ 重要提醒：以上信息仅供健康参考，具体治疗方案和用药请务必遵从医生指导。

回答:""",

        'prevention': """
你是一位专业的医学知识助手。请根据医学参考资料，提供预防和保健建议。

用户问题: {question}

相关医学参考资料:
{context}

表达要求: {expertise_instruction}

请按以下结构回答：

## 🛡️ 预防措施
[具体可行的预防手段]

## 🥗 生活方式建议
[饮食、运动、作息等建议]

## 📅 筛查与随访
[建议的体检/筛查频率，仅当资料中有相关内容时包含]

⚠️ 重要提醒：以上建议仅供参考，个体情况请咨询医生。

回答:""",

        'general': """
你是一位专业的医学知识助手。请根据医学参考资料回答用户的医学问题。

用户问题: {question}

相关医学参考资料:
{context}

表达要求: {expertise_instruction}

请提供准确、清晰的回答：
- 优先使用参考资料中的内容
- 如参考资料不足以回答，请诚实说明
- 涉及诊断治疗的部分提醒就医

回答:""",
    }

    def __init__(self, model_name: str = "qwen3.7-plus", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块

        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("请设置 QWEN_API_KEY 环境变量")

        self.llm = ChatOpenAI(
            model="qwen3.7-plus",  # 或 "qwen-turbo", "qwen-max" 等
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=120,
            max_retries=2
        )

        logger.info("LLM初始化完成")

    # ------------------------------------------------------------------
    # 1. 查询路由 - 使用LLM进行多维度结构化分析
    # ------------------------------------------------------------------
    def query_router(self, query: str) -> Dict[str, Any]:
        """
        查询路由 - 由LLM判断查询类型、紧急程度、专业级别，并提取医学实体

        Args:
            query: 用户查询

        Returns:
            路由信息字典:
            {
                'route': 'emergency|diagnosis|treatment|prevention|general',
                'urgency': 'critical|high|normal',
                'expertise': 'patient|professional',
                'entities': {'symptoms': [], 'diseases': [], 'drugs': [],
                              'body_parts': [], 'examinations': []}
            }
        """
        prompt = ChatPromptTemplate.from_template("""
你是一个医学查询分析助手。请分析用户的医学问题，输出结构化的路由信息。

用户问题: {query}

请分析以下维度：

## 1. 路由类型（route）
- **emergency**: 紧急/危急情况，需要立即急救
  例如：突发剧烈胸痛、大出血、昏迷不醒、中毒、呼吸困难、抽搐
- **diagnosis**: 症状分析/疾病判断
  例如：头痛可能是什么病、这些症状说明什么、糖尿病有哪些症状
- **treatment**: 治疗方案/用药指导
  例如：高血压怎么治疗、阿司匹林用法用量、手术方案
- **prevention**: 预防保健/生活方式
  例如：如何预防感冒、糖尿病饮食注意什么、日常护理
- **general**: 一般科普/概念解释
  例如：什么是高血压、XX和YY的区别、BMI是什么

## 2. 紧急程度（urgency）
- **critical**: 描述危急症状，需要立即就医/急救
- **high**: 建议尽快就医
- **normal**: 常规咨询

## 3. 专业级别（expertise）
- **patient**: 普通患者提问（口语化、非专业表述）
- **professional**: 医学专业人员提问（使用专业术语，提及机制/病理/临床等）

## 4. 医学实体（entities）
提取问题中出现的关键医学信息，没有就填空数组：
- symptoms: 症状
- diseases: 疾病名称
- drugs: 药物名称
- body_parts: 身体部位
- examinations: 检查项目

示例：
- "胸口疼得厉害，喘不上气" → route=emergency, urgency=critical, expertise=patient
- "老是头疼，可能是什么问题" → route=diagnosis, urgency=normal, expertise=patient

## 输出格式（严格输出JSON，不要任何其他内容）
```json
{{
  "route": "emergency|diagnosis|treatment|prevention|general",
  "urgency": "critical|high|normal",
  "expertise": "patient|professional",
  "entities": {{
    "symptoms": [],
    "diseases": [],
    "drugs": [],
    "body_parts": [],
    "examinations": []
  }}
}}
```""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            response = chain.invoke(query).strip()
            parsed = self._parse_json_response(response)
        except Exception as e:
            logger.warning(f"查询路由分析失败，使用默认路由: {e}")
            parsed = {}

        route_info = self._validate_route_info(parsed)
        logger.info(
            f"路由结果: route={route_info['route']}, "
            f"urgency={route_info['urgency']}, "
            f"expertise={route_info['expertise']}"
        )
        return route_info

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM返回的JSON（剥离markdown围栏、提取JSON主体）

        Args:
            response: LLM原始返回文本

        Returns:
            解析后的字典，解析失败返回空字典
        """
        text = response.strip()

        # 剥离 markdown 代码围栏（```json ... ```）
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # 提取第一个 { 到最后一个 } 之间的JSON主体
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            return {}

    def _validate_route_info(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        校验并补全路由信息，非法取值回退到安全默认值

        Args:
            parsed: LLM解析出的路由信息

        Returns:
            完整合法的路由信息字典
        """
        route = parsed.get('route', 'general')
        if route not in self.VALID_ROUTES:
            route = 'general'

        urgency = parsed.get('urgency', 'normal')
        if urgency not in self.VALID_URGENCY:
            urgency = 'normal'

        expertise = parsed.get('expertise', 'patient')
        if expertise not in self.VALID_EXPERTISE:
            expertise = 'patient'

        entities = parsed.get('entities') or {}
        if not isinstance(entities, dict):
            entities = {}
        # 实体字段统一为列表
        for key in ('symptoms', 'diseases', 'drugs', 'body_parts', 'examinations'):
            if not isinstance(entities.get(key), list):
                entities[key] = []

        return {
            'route': route,
            'urgency': urgency,
            'expertise': expertise,
            'entities': entities,
        }

    # ------------------------------------------------------------------
    # 2. 查询重写 - 由LLM判断是否需要重写
    # ------------------------------------------------------------------
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询以提高医学文献检索效果

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能医学查询分析助手。请分析用户的查询，判断是否需要重写以提高医学文献的检索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体疾病/症状名称：如"高血压怎么治疗"、"糖尿病的症状"
   - 明确的用药询问：如"阿司匹林的用法用量"、"二甲双胍的副作用"
   - 具体的检查/治疗询问：如"冠心病的手术方案"、"血常规能查出什么"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"身体不舒服"、"头疼"（无任何限定）
   - 口语化/非规范表述：如"老头晕"、"吃啥药好"
   - 缺少医学限定：如"怎么预防"（未说明预防什么）

重写原则：
- 保持原意不变，不添加用户未提及的具体疾病名
- 将口语化表述转为规范医学表述
- 保持简洁性

示例：
- "高血压怎么治疗" → "高血压怎么治疗"（保持原查询）
- "阿司匹林用法" → "阿司匹林用法"（保持原查询）
- "老头晕是怎么回事" → "反复头晕的常见原因"
- "吃啥药能降血压" → "降血压常用药物"
- "怎么预防" → "常见疾病的预防措施"

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

    # ------------------------------------------------------------------
    # 3. 回答生成 - 统一入口，按路由类型分发到对应模板
    # ------------------------------------------------------------------
    def generate_answer(
        self,
        query: str,
        context_docs: List[Document],
        route_info: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ):
        """
        统一回答生成入口 - 根据路由类型选择回答模板

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            route_info: 路由信息（为空时自动调用query_router分析）
            stream: 是否使用流式输出

        Returns:
            stream=False: 生成的回答字符串
            stream=True: 回答片段生成器
        """
        # 路由分析（未提供时自动分析）
        if route_info is None:
            route_info = self.query_router(query)

        route = route_info.get('route', 'general')
        urgency = route_info.get('urgency', 'normal')
        expertise = route_info.get('expertise', 'patient')

        logger.info(f"生成回答 - 路由: {route}, 紧急程度: {urgency}, 专业级别: {expertise}")

        # 紧急情况使用急救模板（最高优先级）
        if route == 'emergency' or urgency == 'critical':
            template_key = 'emergency'
            prefix = "🚨 **紧急提醒：根据您的描述，这可能属于需要立即处理的紧急情况，请立即拨打 120 或前往急诊！**\n\n"
        else:
            template_key = route if route in self.TEMPLATES else 'general'
            prefix = ""

        if stream:
            return self._generate_stream(query, context_docs, template_key, expertise, prefix)
        return self._generate(query, context_docs, template_key, expertise, prefix)

    def _build_chain(self, template_key: str, context: str, expertise: str):
        """
        使用LCEL构建回答生成链

        Args:
            template_key: 模板键
            context: 已格式化的上下文字符串
            expertise: 专业级别

        Returns:
            LCEL链
        """
        expertise_instruction = self.EXPERTISE_INSTRUCTIONS.get(expertise, self.EXPERTISE_INSTRUCTIONS['patient'])
        prompt = ChatPromptTemplate.from_template(self.TEMPLATES[template_key])

        return (
            {
                "question": RunnablePassthrough(),
                "context": lambda _: context,
                "expertise_instruction": lambda _: expertise_instruction,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _generate(
        self,
        query: str,
        context_docs: List[Document],
        template_key: str,
        expertise: str,
        prefix: str = "",
    ) -> str:
        """非流式生成回答"""
        context = self._build_context(context_docs)
        chain = self._build_chain(template_key, context, expertise)

        response = chain.invoke(query)
        return prefix + response

    def _generate_stream(
        self,
        query: str,
        context_docs: List[Document],
        template_key: str,
        expertise: str,
        prefix: str = "",
    ) -> Generator[str, None, None]:
        """
        流式生成回答

        使用invoke获取完整回答后逐段输出，模拟流式效果
        （避免stream()的SSE流卡死问题，与C8保持一致）
        """
        context = self._build_context(context_docs)
        chain = self._build_chain(template_key, context, expertise)

        try:
            logger.info(f"开始调用LLM生成回答（模板: {template_key}）...")
            response = chain.invoke(query)
            logger.info(f"LLM回答生成完成，长度: {len(response) if response else 0}")

            if prefix:
                yield prefix

            if not response:
                yield "[回答为空]"
                return

            # 逐段输出，模拟流式效果
            chunk_size = 20
            for i in range(0, len(response), chunk_size):
                yield response[i:i + chunk_size]
        except Exception as e:
            logger.error(f"生成出错: {e}")
            yield f"\n[生成中断: {e}]"

    # ------------------------------------------------------------------
    # 4. 上下文构建
    # ------------------------------------------------------------------
    def _build_context(self, docs: List[Document], max_length: int = 6000) -> str:
        """
        构建上下文字符串

        Args:
            docs: 文档列表
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关医学参考资料。"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【资料 {i}】"
            if 'doc_name' in doc.metadata:
                metadata_info += f" {doc.metadata['doc_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 学科: {doc.metadata['category']}"

            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        divider = "\n" + "=" * 50 + "\n"
        return divider + divider.join(context_parts)
