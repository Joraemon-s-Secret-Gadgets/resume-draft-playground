from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Literal

from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings


# =========================================================
# 기본 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "essay_samples.json"
INDEX_DIR = BASE_DIR / "faiss_index"

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "exaone3.5:7.8b"
EMBED_MODEL = "nomic-embed-text"

RETRIEVAL_TOP_K = 5
MIN_SCORE = 0

TARGET_MIN_CHARS = 900
TARGET_MAX_CHARS = 1100
TARGET_PARAGRAPH_MIN = 4
TARGET_PARAGRAPH_MAX = 5

GENERIC_JOBS = {"it", "IT", "개발", "기술", "engineer", "developer"}


# =========================================================
# 데이터 구조
# =========================================================

@dataclass
class DraftRequest:
    applicant_background: str
    target_job: str
    question: str
    constraints: str
    core_experience: str = ""
    strengths: str = ""
    preferred_tone: str = "담백하고 과장 없는 문체"
    banned_points: str = "없는 경험을 추가하지 말 것"


class EssayCase(BaseModel):
    case_id: str
    applicant_background: str
    target_job: str
    question: str
    essay_text: str
    score: int = Field(default=85, ge=0, le=100)
    tags: List[str] = Field(default_factory=list)


class EssayDataset(BaseModel):
    cases: List[EssayCase]


class EssayAnalysis(BaseModel):
    core_theme: str = Field(description="반드시 한국어 한 문장으로 작성. 예시 자소서들이 공통으로 드러내는 중심 주제 1개")
    opening_strategy: List[str] = Field(description="반드시 한국어로 작성. 도입부를 여는 방식 3개")
    paragraph_roles: List[str] = Field(description="반드시 한국어로 작성. 문단별 역할이나 전개 흐름 4~6개")
    reasoning_style: List[str] = Field(description="반드시 한국어로 작성. 문제 인식, 판단, 행동을 연결하는 방식 3개")
    tone_style: List[str] = Field(description="반드시 한국어로 작성. 문체와 톤의 특징 3개")
    persuasion_points: List[str] = Field(description="반드시 한국어로 작성. 설득력이 생기는 핵심 포인트 4개")
    banned_expressions: List[str] = Field(description="반드시 한국어로 작성. 상투적이거나 피해야 할 표현 4개")


class DraftPlan(BaseModel):
    title: str = Field(description="반드시 한국어 한 문장으로 작성. 자소서 전체를 관통하는 한 줄 요약")
    core_message: str = Field(description="반드시 한국어 한 문장으로 작성. 지원자가 가장 강조해야 할 메시지")
    opening_angle: str = Field(description="반드시 한국어 한 문장으로 작성. 도입부에서 어떤 관점으로 시작할지")
    main_experience: str = Field(description="반드시 한국어 한 문장으로 작성. 대표 경험 1개")
    problem_definition: str = Field(description="반드시 한국어 한 문장으로 작성. 경험에서 핵심 문제를 어떻게 볼지")
    actions_and_criteria: List[str] = Field(description="반드시 한국어로 작성. 지원자의 행동과 판단 기준 4~6개")
    learning_points: List[str] = Field(description="반드시 한국어로 작성. 경험을 통해 형성된 관점 3~5개")
    job_connection: List[str] = Field(description="반드시 한국어로 작성. 직무 및 회사와 연결할 포인트 3~4개")
    paragraph_outline: List[str] = Field(description="반드시 한국어로 작성. 문단별 개요 4~5개")


class RevisionOptionSet(BaseModel):
    options: List[str] = Field(description="수정안 3개")


@dataclass
class PendingRevision:
    level: Literal["sentence", "paragraph"]
    target_text: str
    options: List[str]
    instruction: str


@dataclass
class ChatSession:
    request: Optional[DraftRequest] = None
    retrieved_docs: list[Document] = field(default_factory=list)
    analysis: Optional[EssayAnalysis] = None
    plan: Optional[DraftPlan] = None
    current_draft: str = ""
    revision_history: list[str] = field(default_factory=list)
    pending_revision: Optional[PendingRevision] = None


# =========================================================
# 유틸
# =========================================================


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()



def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def contains_job_keyword(text: str, target_job: str) -> bool:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", target_job)
    meaningful_tokens = [t for t in tokens if len(t) >= 2]
    return any(token.lower() in text.lower() for token in meaningful_tokens)



def contains_english_heavily(text: str) -> bool:
    english_chars = re.findall(r"[A-Za-z]", text)
    korean_chars = re.findall(r"[가-힣]", text)
    return len(english_chars) > max(30, len(korean_chars) * 0.3)



def is_placeholder_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    placeholder_patterns = [
        r"^도입 방식 \d+$",
        r"^문단 역할 \d+$",
        r"^판단 전개 방식 \d+$",
        r"^문체 특징 \d+$",
        r"^설득 포인트 \d+$",
        r"^피해야 할 표현 \d+$",
        r"^행동과 판단 기준 \d+$",
        r"^형성된 관점 \d+$",
        r"^직무 연결 포인트 \d+$",
        r"^문단 개요 \d+$",
    ]
    return any(re.match(pattern, stripped) for pattern in placeholder_patterns)



def strip_markdown_headings(text: str) -> str:
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"^\s*---\s*$", "", text, flags=re.MULTILINE)
    return text.strip()



def sanitize_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)

    leading_patterns = [
        r"^Here is .*?:\s*",
        r"^Here’s .*?:\s*",
        r"^Below is .*?:\s*",
        r"^다음은 .*?:\s*",
    ]
    for pattern in leading_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.split(r"\bWord count\s*:\s*\d+\b", cleaned, flags=re.IGNORECASE)[0]
    cleaned = re.split(r"\bChanges made\s*:\s*", cleaned, flags=re.IGNORECASE)[0]
    cleaned = re.split(r"\bRevisions?\s*:\s*", cleaned, flags=re.IGNORECASE)[0]
    cleaned = re.split(r"\bNote\s*:\s*", cleaned, flags=re.IGNORECASE)[0]

    kept_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()

        if not stripped:
            kept_lines.append("")
            continue

        if re.match(r"^\s*#{1,6}\s+", stripped):
            continue

        label_patterns = [
            r"^\[지원자 배경\]$",
            r"^\[지원 직무\]$",
            r"^\[문항\]$",
            r"^\[핵심 문제 정의\]$",
            r"^\[행동과 판단 기준\]$",
            r"^\[형성된 관점\]$",
            r"^\[직무 연결 포인트\]$",
            r"^\[문단 개요\]$",
            r"^\[지원자 이름\]$",
            r"^\(이하 .*작성\)$",
        ]
        if any(re.match(p, stripped, flags=re.IGNORECASE) for p in label_patterns):
            continue

        stripped = stripped.replace("**[지원자 이름]**", "")
        stripped = stripped.replace("[지원자 이름]", "")
        stripped = stripped.replace("**지원자 이름**", "")
        stripped = stripped.replace("지원자 이름", "")

        if re.match(r"^\s*[-*]\s+", stripped):
            continue
        if re.match(r"^\s*\d+\.\s+", stripped):
            continue

        empty_lead_patterns = [
            r"^이러한 문제를 해결하기 위해 다음과 같은 .*:$",
            r"^이 프로젝트를 통해 얻은 핵심 교훈은 다음과 같습니다:?$",
            r"^현재의 경험과 기술을 바탕으로.*다음과 같은 방향으로 기여하고자 합니다:?$",
            r"^다음과 같은 .*취했습니다:?$",
        ]
        if any(re.match(p, stripped) for p in empty_lead_patterns):
            continue

        stripped = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
        stripped = stripped.strip(" :")

        if stripped:
            kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()



def ensure_korean_list(items: List[str], fallback_prefix: str) -> List[str]:
    normalized: list[str] = []
    for idx, item in enumerate(items, start=1):
        text = sanitize_output(item)
        if contains_english_heavily(text) or not text or is_placeholder_text(text):
            text = f"{fallback_prefix} {idx}"
        normalized.append(text)
    return normalized



def infer_tags_from_text(text: str) -> List[str]:
    tags = []
    keyword_map = {
        "AI": ["ai", "인공지능", "llm", "언어 모델"],
        "NLP": ["nlp", "자연어 처리", "huggingface", "transformers"],
        "데이터분석": ["데이터", "분석", "전처리", "시각화"],
        "머신러닝": ["머신러닝", "딥러닝", "pytorch", "모델"],
        "협업": ["팀", "협업", "소통", "프로젝트"],
    }
    lower = text.lower()
    for tag, keys in keyword_map.items():
        if any(k.lower() in lower for k in keys):
            tags.append(tag)
    return tags or ["일반자소서"]



def infer_target_job(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["자연어 처리", "nlp", "llm", "huggingface", "transformers"]):
        return "AI/NLP 직무"
    if any(k in lower for k in ["데이터", "분석", "전처리", "시각화"]):
        return "데이터 분석 직무"
    if any(k in lower for k in ["모델", "머신러닝", "딥러닝", "pytorch"]):
        return "AI/ML 직무"
    return "데이터/AI 직무"



def extract_retrieval_body(raw_text: str) -> str:
    text = strip_markdown_headings(raw_text)

    chunks = re.split(
        r"(?=(?:\d+\)\s*)?(?:자기소개|성격 장단점|기술 역량|지원 동기|입사 후 포부))",
        text,
    )
    chunks = [c.strip() for c in chunks if c.strip()]

    preferred = []
    fallback = []

    for chunk in chunks:
        fallback.append(chunk)
        if any(key in chunk for key in ["자기소개", "성격 장단점", "기술 역량"]):
            preferred.append(chunk)

    selected = preferred if preferred else fallback
    body = "\n\n".join(selected).strip()
    return body[:3500].strip()



def convert_raw_string_to_case(raw_text: str, idx: int) -> EssayCase:
    clean_text = strip_markdown_headings(raw_text)
    retrieval_body = extract_retrieval_body(clean_text)
    tags = infer_tags_from_text(clean_text)
    target_job = infer_target_job(clean_text)

    return EssayCase(
        case_id=f"real_case_{idx:03d}",
        applicant_background="실제 샘플 데이터에서 추출한 자기소개서 예시",
        target_job=target_job,
        question="자기소개서 예시 데이터",
        essay_text=retrieval_body,
        score=85,
        tags=tags,
    )



def build_case_document(case: EssayCase) -> Document:
    page_content = textwrap.dedent(
        f"""
        [지원자 배경]
        {case.applicant_background}

        [지원 직무]
        {case.target_job}

        [문항]
        {case.question}

        [자소서]
        {case.essay_text}
        """
    ).strip()

    return Document(
        page_content=page_content,
        metadata={
            "case_id": case.case_id,
            "target_job": case.target_job,
            "score": case.score,
            "tags": case.tags,
            "essay_text": case.essay_text,
            "question": case.question,
            "applicant_background": case.applicant_background,
        },
    )



def format_analysis(analysis: EssayAnalysis) -> str:
    lines: list[str] = []
    lines.append("[중심 주제]")
    lines.append(analysis.core_theme)
    lines.append("")
    lines.append("[도입 방식]")
    lines.extend([f"- {item}" for item in analysis.opening_strategy])
    lines.append("")
    lines.append("[문단 역할]")
    lines.extend([f"- {item}" for item in analysis.paragraph_roles])
    lines.append("")
    lines.append("[판단 전개 방식]")
    lines.extend([f"- {item}" for item in analysis.reasoning_style])
    lines.append("")
    lines.append("[문체 특징]")
    lines.extend([f"- {item}" for item in analysis.tone_style])
    lines.append("")
    lines.append("[설득 포인트]")
    lines.extend([f"- {item}" for item in analysis.persuasion_points])
    lines.append("")
    lines.append("[피해야 할 표현]")
    lines.extend([f"- {item}" for item in analysis.banned_expressions])
    return "\n".join(lines).strip()



def format_plan(plan: DraftPlan) -> str:
    lines: list[str] = []
    lines.append("[글의 제목]")
    lines.append(plan.title)
    lines.append("")
    lines.append("[핵심 메시지]")
    lines.append(plan.core_message)
    lines.append("")
    lines.append("[도입 관점]")
    lines.append(plan.opening_angle)
    lines.append("")
    lines.append("[대표 경험]")
    lines.append(plan.main_experience)
    lines.append("")
    lines.append("[핵심 문제 정의]")
    lines.append(plan.problem_definition)
    lines.append("")
    lines.append("[행동과 판단 기준]")
    lines.extend([f"- {item}" for item in plan.actions_and_criteria])
    lines.append("")
    lines.append("[형성된 관점]")
    lines.extend([f"- {item}" for item in plan.learning_points])
    lines.append("")
    lines.append("[직무 연결 포인트]")
    lines.extend([f"- {item}" for item in plan.job_connection])
    lines.append("")
    lines.append("[문단 개요]")
    lines.extend([f"- {item}" for item in plan.paragraph_outline])
    return "\n".join(lines).strip()



def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.strip().split("\n\n") if p.strip()]



def infer_input_quality(req: DraftRequest) -> str:
    text = " ".join([
        req.applicant_background,
        req.core_experience,
        req.strengths,
        req.target_job,
        req.question,
    ]).strip()

    if len(text) < 80:
        return "low"
    if req.target_job.strip() in GENERIC_JOBS:
        return "low"
    if len(req.applicant_background.strip()) < 20:
        return "low"
    return "normal"


# =========================================================
# 핵심 서비스
# =========================================================

class ResumeDraftService:
    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        index_dir: Path = INDEX_DIR,
        ollama_base_url: str = OLLAMA_BASE_URL,
        chat_model: str = CHAT_MODEL,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        self.dataset_path = dataset_path
        self.index_dir = index_dir

        self.llm = ChatOllama(
            model=chat_model,
            base_url=ollama_base_url,
            temperature=0.15,
        )

        self.analysis_llm = self.llm.with_structured_output(EssayAnalysis)
        self.plan_llm = self.llm.with_structured_output(DraftPlan)
        self.revision_llm = self.llm.with_structured_output(RevisionOptionSet)

        self.embeddings = OllamaEmbeddings(
            model=embed_model,
            base_url=ollama_base_url,
        )

        self.dataset = self._load_dataset()
        self.vectorstore = self._load_or_build_vectorstore()

        self.retrieve_chain = RunnableLambda(self.retrieve_examples)
        self.analyze_chain = RunnableLambda(self.analyze_examples)
        self.plan_chain = RunnableLambda(self.plan_draft)
        self.generate_chain = RunnableLambda(self.generate_draft)
        self.expand_chain = RunnableLambda(self.expand_draft_if_short)
        self.rewrite_chain = RunnableLambda(self.rewrite_draft)

    def _load_dataset(self) -> EssayDataset:
        raw = read_json(self.dataset_path)

        if isinstance(raw, dict) and "cases" in raw:
            return EssayDataset.model_validate(raw)

        if isinstance(raw, list):
            cases: list[EssayCase] = []

            for idx, item in enumerate(raw, start=1):
                if isinstance(item, dict):
                    cases.append(
                        EssayCase(
                            case_id=item.get("case_id", f"real_case_{idx:03d}"),
                            applicant_background=item.get("applicant_background", "실제 샘플 데이터에서 추출한 자기소개서 예시"),
                            target_job=item.get("target_job", infer_target_job(json.dumps(item, ensure_ascii=False))),
                            question=item.get("question", "자기소개서 예시 데이터"),
                            essay_text=item.get("essay_text", json.dumps(item, ensure_ascii=False)),
                            score=int(item.get("score", 85)),
                            tags=item.get("tags", infer_tags_from_text(json.dumps(item, ensure_ascii=False))),
                        )
                    )
                elif isinstance(item, str):
                    cases.append(convert_raw_string_to_case(item, idx))
                else:
                    continue

            return EssayDataset(cases=cases)

        raise ValueError("지원하지 않는 essay_samples.json 구조입니다.")

    def _load_or_build_vectorstore(self) -> FAISS:
        if self.index_dir.exists():
            try:
                return FAISS.load_local(
                    folder_path=str(self.index_dir),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                pass

        documents = [build_case_document(case) for case in self.dataset.cases]
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.index_dir))
        return vectorstore

    def _normalize_analysis(self, analysis: EssayAnalysis) -> EssayAnalysis:
        return EssayAnalysis(
            core_theme=(
                sanitize_output(analysis.core_theme)
                if not contains_english_heavily(analysis.core_theme) and not is_placeholder_text(analysis.core_theme)
                else "지원자의 경험을 문제 인식과 역량으로 연결하는 자기소개서 구조"
            ),
            opening_strategy=ensure_korean_list(analysis.opening_strategy[:3], "도입 방식"),
            paragraph_roles=ensure_korean_list(analysis.paragraph_roles[:6], "문단 역할"),
            reasoning_style=ensure_korean_list(analysis.reasoning_style[:3], "판단 전개 방식"),
            tone_style=ensure_korean_list(analysis.tone_style[:3], "문체 특징"),
            persuasion_points=ensure_korean_list(analysis.persuasion_points[:4], "설득 포인트"),
            banned_expressions=ensure_korean_list(analysis.banned_expressions[:4], "피해야 할 표현"),
        )

    def _normalize_plan(self, plan: DraftPlan, req: DraftRequest | None = None) -> DraftPlan:
        default_title = "기준과 구조를 먼저 생각하는 지원자"
        default_core_message = "주어진 정보를 정리하고 활용 가능한 형태로 만드는 역량"
        default_opening = "결과보다 기준과 흐름을 먼저 생각하는 태도에서 시작"
        default_main_experience = req.core_experience.strip() if req and req.core_experience.strip() else "사용자가 직접 입력한 대표 경험"
        default_problem = "주어진 경험 안에서 무엇이 핵심 문제였는지 분명히 보고 정리하는 태도"

        title = sanitize_output(plan.title)
        core_message = sanitize_output(plan.core_message)
        opening_angle = sanitize_output(plan.opening_angle)
        main_experience = sanitize_output(plan.main_experience)
        problem_definition = sanitize_output(plan.problem_definition)

        if contains_english_heavily(title) or not title or is_placeholder_text(title):
            title = default_title
        if contains_english_heavily(core_message) or not core_message or is_placeholder_text(core_message):
            core_message = default_core_message
        if contains_english_heavily(opening_angle) or not opening_angle or is_placeholder_text(opening_angle):
            opening_angle = default_opening
        if contains_english_heavily(main_experience) or not main_experience or is_placeholder_text(main_experience):
            main_experience = default_main_experience
        if contains_english_heavily(problem_definition) or not problem_definition or is_placeholder_text(problem_definition):
            problem_definition = default_problem

        actions_and_criteria = ensure_korean_list(plan.actions_and_criteria[:6], "행동과 판단 기준")
        learning_points = ensure_korean_list(plan.learning_points[:5], "형성된 관점")
        job_connection = ensure_korean_list(plan.job_connection[:4], "직무 연결 포인트")
        paragraph_outline = ensure_korean_list(plan.paragraph_outline[:5], "문단 개요")

        return DraftPlan(
            title=title,
            core_message=core_message,
            opening_angle=opening_angle,
            main_experience=main_experience,
            problem_definition=problem_definition,
            actions_and_criteria=actions_and_criteria,
            learning_points=learning_points,
            job_connection=job_connection,
            paragraph_outline=paragraph_outline,
        )

    def retrieve_examples(self, req: DraftRequest) -> list[Document]:
        query = textwrap.dedent(
            f"""
            지원자 배경: {req.applicant_background}
            대표 경험: {req.core_experience}
            지원 직무: {req.target_job}
            문항: {req.question}
            """
        ).strip()

        candidates = self.vectorstore.similarity_search(query, k=15)

        ranked: list[tuple[int, int, int, Document]] = []
        for doc in candidates:
            score = int(doc.metadata.get("score", 85))
            job_text = normalize_text(str(doc.metadata.get("target_job", "")))
            tags = " ".join(doc.metadata.get("tags", []))
            essay_text = str(doc.metadata.get("essay_text", ""))

            same_job_bonus = 1 if job_text == normalize_text(req.target_job) else 0
            weak_match_bonus = 1 if contains_job_keyword(job_text + " " + tags + " " + essay_text, req.target_job) else 0

            ranked.append((same_job_bonus, weak_match_bonus, score, doc))

        ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        selected = [doc for _, _, _, doc in ranked[:RETRIEVAL_TOP_K]]
        if not selected:
            selected = candidates[:RETRIEVAL_TOP_K]
        return selected

    def analyze_examples(self, inputs: dict[str, Any]) -> EssayAnalysis:
        docs: list[Document] = inputs["retrieved_docs"]

        examples_text = "\n\n".join(
            [
                textwrap.dedent(
                    f"""
                    [예시 자소서 {idx}]
                    추정 직무: {doc.metadata.get("target_job")}
                    태그: {", ".join(doc.metadata.get("tags", []))}
                    자소서:
                    {doc.metadata.get("essay_text")}
                    """
                ).strip()
                for idx, doc in enumerate(docs, start=1)
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    너는 한국어 자기소개서 분석 시스템이다.
                    반드시 한국어로만 작성할 것.
                    역할은 참고 자소서들에서 문장 자체가 아니라 자기소개서의 전개 방식과 설득 구조를 추출하는 것이다.
                    모든 필드는 한국어 짧은 문장으로 작성할 것.
                    """.strip(),
                ),
                (
                    "human",
                    """
                    아래 참고 자소서들을 분석해줘.

                    {examples_text}

                    분석할 내용:
                    - 글 전체를 관통하는 핵심 주제
                    - 도입부를 여는 방식
                    - 자소서에서 문단들이 어떤 역할로 배치되는지
                    - 문제 인식 → 판단 → 행동 → 의미 연결 방식
                    - 문체와 톤의 특징
                    - 설득력이 생기는 핵심 포인트
                    - 피해야 할 상투적 표현
                    """.strip(),
                ),
            ]
        )

        chain = prompt | self.analysis_llm
        result = chain.invoke({"examples_text": examples_text})
        return self._normalize_analysis(result)

    def plan_draft(self, inputs: dict[str, Any]) -> DraftPlan:
        req: DraftRequest = inputs["request"]
        analysis: EssayAnalysis = inputs["analysis"]
        quality = infer_input_quality(req)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    너는 한국어 자기소개서 설계 전문가다.
                    반드시 한국어로만 작성할 것.

                    역할은 지원자의 실제 배경과 예시 분석 결과를 바탕으로 자기소개서의 설계안을 만드는 것이다.

                    반드시 지켜야 할 원칙:
                    1. 지원자가 제공하지 않은 경험을 새로 만들지 말 것.
                    2. 예시 데이터의 표면 문장보다 전개 구조만 참고할 것.
                    3. 제목형 목차를 만들지 말고, 실제 본문 흐름을 설계할 것.
                    4. 정보가 부족하면 일반론으로 부풀리지 말고, 현재 정보 범위 안에서 정직하게 설계할 것.
                    """.strip(),
                ),
                (
                    "human",
                    """
                    아래 정보를 바탕으로 자기소개서 설계안을 만들어줘.

                    [지원자 배경]
                    {applicant_background}

                    [대표 경험]
                    {core_experience}

                    [강점]
                    {strengths}

                    [지원 직무]
                    {target_job}

                    [문항]
                    {question}

                    [작성 조건]
                    {constraints}

                    [선호 문체]
                    {preferred_tone}

                    [금지 사항]
                    {banned_points}

                    [입력 정보 충분도]
                    {quality}

                    [예시 분석 결과]
                    {analysis_text}
                    """.strip(),
                ),
            ]
        )

        chain = prompt | self.plan_llm
        result = chain.invoke(
            {
                "applicant_background": req.applicant_background,
                "core_experience": req.core_experience,
                "strengths": req.strengths,
                "target_job": req.target_job,
                "question": req.question,
                "constraints": req.constraints,
                "preferred_tone": req.preferred_tone,
                "banned_points": req.banned_points,
                "quality": quality,
                "analysis_text": format_analysis(analysis),
            }
        )
        return self._normalize_plan(result, req=req)

    def generate_draft(self, inputs: dict[str, Any]) -> str:
        req: DraftRequest = inputs["request"]
        analysis: EssayAnalysis = inputs["analysis"]
        plan: DraftPlan = inputs["plan"]
        quality = infer_input_quality(req)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    너는 한국어 자기소개서 작성 전문가다.
                    반드시 한국어로만 작성할 것.

                    반드시 지켜야 할 원칙:
                    1. 자기소개서 본문만 작성할 것.
                    2. 지원자가 제공하지 않은 경험, 성과, 수치, 역할은 절대 새로 만들지 말 것.
                    3. 인턴십, 프로젝트, 사용자 행동 분석, 고객 문의 데이터 같은 표현을 사용자가 제공하지 않았다면 쓰지 말 것.
                    4. 제목, 소제목, 마크다운 헤더를 절대 출력하지 말 것.
                    5. '[지원자 이름]' 같은 placeholder를 절대 출력하지 말 것.
                    6. '다음과 같은', '핵심 교훈은 다음과 같습니다'처럼 뒤 내용 없이 끝나는 예고문을 쓰지 말 것.
                    7. 같은 표현을 반복해 분량을 채우지 말 것.
                    8. 과장된 표현(혁신가, 전문가, 최고의, 중추적 역할)을 남발하지 말 것.
                    9. 입력 정보가 부족하면 정직하게 범위 안에서만 작성하고, 없는 경험을 만들어서 분량을 채우지 말 것.
                    10. 지원 직무가 포괄적이더라도 사용자가 준 정보 안에서만 연결할 것.

                    작성 기준:
                    - 전체 분량은 {TARGET_MIN_CHARS}~{TARGET_MAX_CHARS}자 내외
                    - 문단 수는 {TARGET_PARAGRAPH_MIN}~{TARGET_PARAGRAPH_MAX}문단
                    - 각 문단은 자연스러운 본문 형태로만 작성
                    """.strip(),
                ),
                (
                    "human",
                    """
                    아래 정보를 바탕으로 자기소개서를 작성해줘.

                    [지원자 배경]
                    {applicant_background}

                    [대표 경험]
                    {core_experience}

                    [강점]
                    {strengths}

                    [지원 직무]
                    {target_job}

                    [문항]
                    {question}

                    [작성 조건]
                    {constraints}

                    [선호 문체]
                    {preferred_tone}

                    [금지 사항]
                    {banned_points}

                    [입력 정보 충분도]
                    {quality}

                    [예시 분석 결과]
                    {analysis_text}

                    [설계안]
                    {plan_text}

                    작성 지시:
                    - 도입부에서는 지원자의 일하는 기준이나 문제를 보는 태도가 자연스럽게 드러나게 할 것
                    - 대표 경험은 사용자가 입력한 경험만 활용할 것
                    - 경험이 부족하면 없는 사례를 만들지 말고 현재 정보 범위에서 정직하게 서술할 것
                    - 마지막은 지원 직무와의 연결로 마무리할 것
                    - 소제목 없이 자기소개서 본문만 출력할 것
                    """.strip(),
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke(
            {
                "applicant_background": req.applicant_background,
                "core_experience": req.core_experience,
                "strengths": req.strengths,
                "target_job": req.target_job,
                "question": req.question,
                "constraints": req.constraints,
                "preferred_tone": req.preferred_tone,
                "banned_points": req.banned_points,
                "quality": quality,
                "analysis_text": format_analysis(analysis),
                "plan_text": format_plan(plan),
            }
        )
        return sanitize_output(raw)

    def expand_draft_if_short(self, inputs: dict[str, Any]) -> str:
        req: DraftRequest = inputs["request"]
        short_draft: str = sanitize_output(inputs["short_draft"])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    너는 한국어 자기소개서 확장 전문가다.
                    반드시 한국어로만 작성할 것.
                    제목, 소제목, 마크다운 헤더를 넣지 말 것.
                    기존 초안의 흐름을 유지하면서 {TARGET_MIN_CHARS}~{TARGET_MAX_CHARS}자 내외가 되도록 자연스럽게 확장할 것.
                    지원자가 제공하지 않은 경험은 절대 새로 만들지 말 것.
                    """.strip(),
                ),
                (
                    "human",
                    """
                    아래 초안은 전체 흐름은 괜찮지만 글자 수가 부족하다.
                    부족한 부분만 자연스럽게 확장해줘.

                    [현재 초안]
                    {short_draft}

                    [지원자 배경]
                    {applicant_background}

                    [대표 경험]
                    {core_experience}
                    """.strip(),
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke(
            {
                "short_draft": short_draft,
                "applicant_background": req.applicant_background,
                "core_experience": req.core_experience,
            }
        )
        return sanitize_output(raw)

    def rewrite_draft(self, inputs: dict[str, Any]) -> str:
        bad_draft: str = sanitize_output(inputs["bad_draft"])
        reason: str = inputs["reason"]
        req: DraftRequest = inputs["request"]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    너는 한국어 자기소개서 리라이트 전문가다.
                    반드시 한국어로만 작성할 것.
                    제목, 소제목, 마크다운 헤더를 넣지 말 것.
                    placeholder를 넣지 말 것.
                    {TARGET_MIN_CHARS}~{TARGET_MAX_CHARS}자 내외를 유지할 것.
                    지원자가 제공하지 않은 경험은 절대 새로 만들지 말 것.
                    """.strip(),
                ),
                (
                    "human",
                    """
                    아래 초안을 고쳐줘.

                    [수정 이유]
                    {reason}

                    [이전 초안]
                    {bad_draft}

                    [지원자 배경]
                    {applicant_background}

                    [대표 경험]
                    {core_experience}

                    수정 지시:
                    - 핵심 내용은 유지할 것
                    - 반복과 과한 표현을 줄일 것
                    - 자기소개서 본문만 출력할 것
                    """.strip(),
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke(
            {
                "reason": reason,
                "bad_draft": bad_draft,
                "applicant_background": req.applicant_background,
                "core_experience": req.core_experience,
            }
        )
        return sanitize_output(raw)

    def validate_draft(self, text: str, req: DraftRequest) -> tuple[bool, str]:
        stripped = sanitize_output(text)

        if contains_english_heavily(stripped):
            return False, "영어 비중이 너무 높음"

        banned_patterns = [
            r"^\s*#{1,6}\s+",
            r"\[지원자 이름\]",
            r"지원자 이름",
            r"다음과 같은 .*:$",
            r"핵심 교훈은 다음과 같습니다:?$",
            r"방향으로 기여하고자 합니다:?$",
        ]
        for pattern in banned_patterns:
            if re.search(pattern, stripped, flags=re.MULTILINE):
                return False, f"금지된 형식 포함: {pattern}"

        paragraphs = split_paragraphs(stripped)
        if len(paragraphs) < TARGET_PARAGRAPH_MIN:
            return False, f"문단 수가 너무 적음: {len(paragraphs)}개"
        if len(paragraphs) > TARGET_PARAGRAPH_MAX:
            return False, f"문단 수가 너무 많음: {len(paragraphs)}개"

        if len(stripped) < TARGET_MIN_CHARS - 50:
            return False, "본문이 너무 짧음: 문제 정의·판단 기준·직무 연결 보강 필요"
        if len(stripped) > TARGET_MAX_CHARS + 100:
            return False, "본문이 너무 김"

        repetitive_phrases = [
            "이러한 경험을 통해",
            "실제 의사결정에 연결",
            "흥미를 느끼게 되었습니다",
            "흥미를 느끼게 되었다",
            "문제 인식과 판단 과정",
        ]
        for phrase in repetitive_phrases:
            if stripped.count(phrase) >= 3:
                return False, f"반복 표현이 많음: {phrase}"

        overblown_phrases = [
            "혁신가",
            "전문가",
            "중추적인 역할",
            "최고의",
            "완벽한",
        ]
        if sum(stripped.count(p) for p in overblown_phrases) >= 2:
            return False, "과한 표현이 많음"

        suspicious_invented_phrases = [
            "인턴십 기간 동안",
            "고객 문의 데이터",
            "사용자 행동 분석 프로젝트",
            "웹사이트 방문자",
            "이탈률을 감소",
        ]
        for phrase in suspicious_invented_phrases:
            if phrase in stripped and phrase not in req.applicant_background and phrase not in req.core_experience:
                return False, f"입력되지 않은 경험이 포함됨: {phrase}"

        last_one_or_two = "\n\n".join(paragraphs[-2:])
        if not contains_job_keyword(last_one_or_two, req.target_job):
            return False, "후반부에서 지원 직무와의 연결이 약함"

        return True, "통과"

    def run(self, req: DraftRequest) -> dict[str, Any]:
        retrieved_docs = self.retrieve_examples(req)
        analysis = self.analyze_examples({"request": req, "retrieved_docs": retrieved_docs})
        plan = self.plan_draft({"request": req, "analysis": analysis})
        first_draft = sanitize_output(self.generate_draft({"request": req, "analysis": analysis, "plan": plan}))

        valid, reason = self.validate_draft(first_draft, req)
        if valid:
            return {
                "retrieved_docs": retrieved_docs,
                "analysis": analysis,
                "plan": plan,
                "final_draft": first_draft,
                "valid": True,
                "reason": reason,
            }

        expanded_draft = first_draft
        expanded_reason = reason
        if "본문이 너무 짧음" in reason:
            expanded_draft = sanitize_output(
                self.expand_draft_if_short(
                    {
                        "request": req,
                        "analysis": analysis,
                        "plan": plan,
                        "short_draft": first_draft,
                    }
                )
            )
            expanded_valid, expanded_reason = self.validate_draft(expanded_draft, req)
            if expanded_valid:
                return {
                    "retrieved_docs": retrieved_docs,
                    "analysis": analysis,
                    "plan": plan,
                    "first_draft": first_draft,
                    "expanded_draft": expanded_draft,
                    "final_draft": expanded_draft,
                    "valid": True,
                    "reason": expanded_reason,
                }

        rewritten = sanitize_output(
            self.rewrite_draft(
                {
                    "request": req,
                    "analysis": analysis,
                    "plan": plan,
                    "bad_draft": expanded_draft,
                    "reason": expanded_reason,
                }
            )
        )
        valid2, reason2 = self.validate_draft(rewritten, req)

        return {
            "retrieved_docs": retrieved_docs,
            "analysis": analysis,
            "plan": plan,
            "first_draft": first_draft,
            "expanded_draft": expanded_draft if expanded_draft != first_draft else None,
            "final_draft": rewritten,
            "valid": valid2,
            "reason": reason2,
        }

    # =====================================================
    # 대화형 수정 기능
    # =====================================================

    def suggest_sentence_revisions(
        self,
        current_draft: str,
        target_sentence: str,
        user_request: str,
    ) -> List[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    너는 한국어 자기소개서 문장 편집 도우미다.
                    반드시 한국어로만 작성할 것.
                    사용자가 수정하고 싶은 한 문장에 대해, 서로 다른 수정안 3개를 제시할 것.
                    의미는 유지하되 톤과 표현을 다르게 제안할 것.
                    번호, bullet 없이 수정안 문장 3개만 반환하라.
                    """.strip(),
                ),
                (
                    "human",
                    """
                    [현재 초안]
                    {current_draft}

                    [수정 대상 문장]
                    {target_sentence}

                    [수정 요청]
                    {user_request}

                    수정안 3개를 제시해줘.
                    """.strip(),
                ),
            ]
        )

        raw = (prompt | self.revision_llm).invoke(
            {
                "current_draft": current_draft,
                "target_sentence": target_sentence,
                "user_request": user_request,
            }
        )
        return [sanitize_output(x) for x in raw.options[:3]]

    def suggest_paragraph_revisions(
        self,
        current_draft: str,
        target_paragraph: str,
        user_request: str,
    ) -> List[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    너는 한국어 자기소개서 문단 편집 도우미다.
                    반드시 한국어로만 작성할 것.
                    사용자가 수정하고 싶은 한 문단에 대해, 서로 다른 수정안 3개를 제시할 것.
                    의미와 흐름은 유지하되 톤, 길이, 표현을 다르게 조정할 것.
                    번호, bullet 없이 문단 수정안 3개만 반환하라.
                    """.strip(),
                ),
                (
                    "human",
                    """
                    [현재 초안]
                    {current_draft}

                    [수정 대상 문단]
                    {target_paragraph}

                    [수정 요청]
                    {user_request}

                    수정안 3개를 제시해줘.
                    """.strip(),
                ),
            ]
        )

        raw = (prompt | self.revision_llm).invoke(
            {
                "current_draft": current_draft,
                "target_paragraph": target_paragraph,
                "user_request": user_request,
            }
        )
        return [sanitize_output(x) for x in raw.options[:3]]

    def revise_full_draft_by_feedback(
        self,
        current_draft: str,
        user_request: str,
        req: DraftRequest,
    ) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    너는 한국어 자기소개서 전체 수정 도우미다.
                    반드시 한국어로만 작성할 것.

                    원칙:
                    1. 기존 초안의 핵심 경험과 흐름은 유지할 것.
                    2. 사용자의 수정 요청만 반영할 것.
                    3. 제목, 소제목, bullet, placeholder를 넣지 말 것.
                    4. 분량은 {TARGET_MIN_CHARS}~{TARGET_MAX_CHARS}자 내외 유지.
                    5. 문단 수는 {TARGET_PARAGRAPH_MIN}~{TARGET_PARAGRAPH_MAX}문단 유지.
                    6. 사용자가 제공하지 않은 경험은 새로 만들지 말 것.
                    """.strip(),
                ),
                (
                    "human",
                    """
                    [현재 초안]
                    {current_draft}

                    [지원 직무]
                    {target_job}

                    [대표 경험]
                    {core_experience}

                    [사용자 수정 요청]
                    {user_request}

                    현재 초안을 사용자의 요청에 맞게 전체 수정해줘.
                    """.strip(),
                ),
            ]
        )

        raw = (prompt | self.llm | StrOutputParser()).invoke(
            {
                "current_draft": current_draft,
                "target_job": req.target_job,
                "core_experience": req.core_experience,
                "user_request": user_request,
            }
        )
        return sanitize_output(raw)

    def apply_selected_revision(
        self,
        current_draft: str,
        target_text: str,
        selected_option: str,
    ) -> str:
        if target_text not in current_draft:
            return current_draft
        return current_draft.replace(target_text, selected_option, 1)


# =========================================================
# 사용자 입력
# =========================================================

def collect_user_request() -> DraftRequest:
    print("\n고객 정보를 입력해주세요.")
    applicant_background = input("1) 배경/경험: ").strip()
    target_job = input("2) 지원 직무: ").strip()
    question = input("3) 문항: ").strip()
    constraints = input("4) 작성 조건(없으면 엔터): ").strip()
    core_experience = input("5) 꼭 넣고 싶은 대표 경험(없으면 엔터): ").strip()
    strengths = input("6) 강조할 강점/키워드(없으면 엔터): ").strip()
    preferred_tone = input("7) 선호 문체(없으면 엔터): ").strip()
    banned_points = input("8) 넣지 말아야 할 내용/톤(없으면 엔터): ").strip()

    if not constraints:
        constraints = "1000자 내외 자기소개서, 과장 없는 문체, 직무 적합성과 일하는 방식이 드러나게 작성"
    if not preferred_tone:
        preferred_tone = "담백하고 과장 없는 문체"
    if not banned_points:
        banned_points = "없는 경험을 추가하지 말 것"

    return DraftRequest(
        applicant_background=applicant_background,
        target_job=target_job,
        question=question,
        constraints=constraints,
        core_experience=core_experience,
        strengths=strengths,
        preferred_tone=preferred_tone,
        banned_points=banned_points,
    )



def print_help() -> None:
    print(
        textwrap.dedent(
            """
            사용 가능한 명령:
            - 초안 생성
            - 보기
            - 문장 수정: <원문 문장> | <수정 요청>
            - 문단 수정: <문단번호> | <수정 요청>
            - 전체 수정: <수정 요청>
            - 적용: <번호>
            - 처음부터
            - 도움말
            - 종료
            """
        ).strip()
    )


# =========================================================
# 실행
# =========================================================

def main() -> None:
    print("=" * 80)
    print("유사 사례 기반 자기소개서 챗봇")
    print("=" * 80)
    print("RUNNING FILE:", __file__)
    print("CHAT_MODEL:", CHAT_MODEL)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"데이터셋 파일이 없습니다: {DATASET_PATH}")

    service = ResumeDraftService()
    print("SERVICE MODEL:", service.llm.model)
    print("DATASET SIZE:", len(service.dataset.cases))

    session = ChatSession()
    print_help()

    while True:
        user_input = input("\n사용자> ").strip()

        if not user_input:
            continue

        if user_input == "종료":
            print("챗봇을 종료합니다.")
            break

        if user_input == "도움말":
            print_help()
            continue

        if user_input == "처음부터":
            session = ChatSession()
            print("세션을 초기화했습니다. 다시 초안을 생성해주세요.")
            continue

        if user_input == "초안 생성":
            session.request = collect_user_request()
            result = service.run(session.request)

            session.retrieved_docs = result["retrieved_docs"]
            session.analysis = result["analysis"]
            session.plan = result["plan"]
            session.current_draft = result["final_draft"]
            session.revision_history = ["초안 생성"]
            session.pending_revision = None

            print("\n[생성된 초안]")
            print("-" * 80)
            print(session.current_draft)
            print("\n[검사 결과]")
            print(result["reason"])
            continue

        if user_input == "보기":
            if not session.current_draft:
                print("아직 초안이 없습니다. 먼저 '초안 생성'을 입력하세요.")
                continue

            print("\n[현재 초안]")
            print("-" * 80)
            print(session.current_draft)
            continue

        if user_input.startswith("문장 수정:"):
            if not session.current_draft:
                print("먼저 초안을 생성하세요.")
                continue

            payload = user_input.replace("문장 수정:", "", 1).strip()
            if "|" not in payload:
                print("형식: 문장 수정: <원문 문장> | <수정 요청>")
                continue

            target_sentence, instruction = [x.strip() for x in payload.split("|", 1)]

            if target_sentence not in session.current_draft:
                print("현재 초안에서 해당 문장을 찾지 못했습니다. 문장을 그대로 복사해서 넣어주세요.")
                continue

            options = service.suggest_sentence_revisions(
                current_draft=session.current_draft,
                target_sentence=target_sentence,
                user_request=instruction,
            )

            session.pending_revision = PendingRevision(
                level="sentence",
                target_text=target_sentence,
                options=options,
                instruction=instruction,
            )

            print("\n[문장 수정안]")
            for idx, option in enumerate(options, start=1):
                print(f"{idx}. {option}")
            print("원하는 안을 선택하려면: 적용: 1")
            continue

        if user_input.startswith("문단 수정:"):
            if not session.current_draft:
                print("먼저 초안을 생성하세요.")
                continue

            payload = user_input.replace("문단 수정:", "", 1).strip()
            if "|" not in payload:
                print("형식: 문단 수정: <문단번호> | <수정 요청>")
                continue

            paragraph_no_str, instruction = [x.strip() for x in payload.split("|", 1)]
            if not paragraph_no_str.isdigit():
                print("문단번호는 숫자로 입력해주세요.")
                continue

            paragraphs = split_paragraphs(session.current_draft)
            paragraph_no = int(paragraph_no_str)

            if paragraph_no < 1 or paragraph_no > len(paragraphs):
                print(f"문단번호 범위를 벗어났습니다. 현재 문단 수: {len(paragraphs)}")
                continue

            target_paragraph = paragraphs[paragraph_no - 1]
            options = service.suggest_paragraph_revisions(
                current_draft=session.current_draft,
                target_paragraph=target_paragraph,
                user_request=instruction,
            )

            session.pending_revision = PendingRevision(
                level="paragraph",
                target_text=target_paragraph,
                options=options,
                instruction=instruction,
            )

            print("\n[문단 수정안]")
            for idx, option in enumerate(options, start=1):
                print(f"{idx}. {option}\n")
            print("원하는 안을 선택하려면: 적용: 1")
            continue

        if user_input.startswith("전체 수정:"):
            if not session.current_draft or not session.request:
                print("먼저 초안을 생성하세요.")
                continue

            instruction = user_input.replace("전체 수정:", "", 1).strip()
            updated = service.revise_full_draft_by_feedback(
                current_draft=session.current_draft,
                user_request=instruction,
                req=session.request,
            )

            session.current_draft = updated
            session.revision_history.append(f"전체 수정: {instruction}")
            session.pending_revision = None

            print("\n[전체 수정 반영본]")
            print("-" * 80)
            print(session.current_draft)

            valid, reason = service.validate_draft(session.current_draft, session.request)
            print("\n[검사 결과]")
            print(reason)
            continue

        if user_input.startswith("적용:"):
            if not session.pending_revision:
                print("현재 선택 가능한 수정안이 없습니다.")
                continue

            choice_str = user_input.replace("적용:", "", 1).strip()
            if not choice_str.isdigit():
                print("형식: 적용: 1")
                continue

            choice = int(choice_str)
            if choice < 1 or choice > len(session.pending_revision.options):
                print("선택 번호가 범위를 벗어났습니다.")
                continue

            selected_option = session.pending_revision.options[choice - 1]
            session.current_draft = service.apply_selected_revision(
                current_draft=session.current_draft,
                target_text=session.pending_revision.target_text,
                selected_option=selected_option,
            )
            session.revision_history.append(
                f"{session.pending_revision.level} 수정 반영: {session.pending_revision.instruction} / option {choice}"
            )
            session.pending_revision = None

            print("\n[반영된 초안]")
            print("-" * 80)
            print(session.current_draft)

            if session.request:
                valid, reason = service.validate_draft(session.current_draft, session.request)
                print("\n[검사 결과]")
                print(reason)
            continue

        print("알 수 없는 명령입니다. '도움말'을 입력해 형식을 확인하세요.")


if __name__ == "__main__":
    main()
