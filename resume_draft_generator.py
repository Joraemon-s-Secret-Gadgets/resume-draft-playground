from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import List, Optional

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
TIMEOUT_SECONDS = 300


@dataclass
class DraftRequest:
    applicant_background: str
    target_job: str
    question: str
    constraints: str
    example_essays: List[str]


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def generate_draft(self, req: DraftRequest) -> str:
        system_prompt = textwrap.dedent(
            """
            너는 한국어 자기소개서 초안 작성 도우미다.
            역할은 '예시 자소서의 구조와 장점'을 참고하되,
            문장을 베끼지 않고 지원자의 실제 이력에 맞춰 새로운 초안을 작성하는 것이다.

            반드시 지켜야 할 원칙:
            1. 예시 자소서의 문장과 표현을 그대로 복사하지 말 것.
            2. 구조, 전개 방식, 강조 포인트만 참고할 것.
            3. 지원자가 제공하지 않은 경험을 지어내지 말 것.
            4. 한국어로 자연스럽고 담백하게 작성할 것.
            5. 지나치게 과장된 표현, 추상적인 미사여구를 줄일 것.
            6. 문항에 대한 직접적인 답이 되도록 작성할 것.
            7. 결과는 하나의 완성된 초안으로 작성할 것.
            8. 마지막에 '활용한 강조 포인트'를 3개 bullet로 요약할 것.
            """
        ).strip()

        examples_text = "\n\n".join(
            [f"[예시 자소서 {idx}]\n{essay.strip()}" for idx, essay in enumerate(req.example_essays, start=1)]
        )

        user_prompt = textwrap.dedent(
            f"""
            아래 정보를 바탕으로 자기소개서 초안을 작성해줘.

            [지원자 배경]
            {req.applicant_background.strip()}

            [지원 직무]
            {req.target_job.strip()}

            [문항]
            {req.question.strip()}

            [작성 조건]
            {req.constraints.strip()}

            [참고할 예시 자소서들]
            {examples_text}

            출력 형식:
            1. 제목 없이 바로 초안 본문 작성
            2. 본문 아래에 빈 줄 한 줄
            3. '활용한 강조 포인트'라는 소제목 작성
            4. bullet 3개로 정리
            """
        ).strip()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "keep_alive": "10m",
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama 호출 실패: {exc}") from exc

        data = response.json()
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"응답 형식이 예상과 다릅니다: {json.dumps(data, ensure_ascii=False, indent=2)}") from exc


def load_sample_request() -> DraftRequest:
    return DraftRequest(
        applicant_background=(
            "광운대학교 정보융합학과 재학 중이며, 데이터 분석과 자동화에 관심이 많다. "
            "Python과 MySQL로 데이터를 수집·전처리했고, Excel과 Tableau로 시각화한 경험이 있다. "
            "학부 프로젝트로 뉴스와 공시 데이터를 활용한 ESG 포트폴리오 추천 서비스를 진행하며 "
            "데이터 구조 설계와 전처리 기준 정리에 참여했다. 반복 업무를 줄이고, 데이터를 실제 의사결정에 연결하는 일에 흥미를 느낀다."
        ),
        target_job="데이터 분석 직무",
        question="지원 직무에 적합한 본인의 강점과 이를 보여주는 경험을 서술해 주세요.",
        constraints="700자 내외, 과장 없는 문체, 실무 적용 가능성이 드러나게 작성",
        example_essays=[
            (
                "저는 문제를 직관이 아니라 데이터로 확인하는 습관을 길러 왔습니다. 학교 프로젝트에서 여러 출처의 데이터를 정리하는 과정에서 "
                "형식이 맞지 않아 분석이 지연되는 문제를 겪었고, 이를 해결하기 위해 기준 컬럼을 다시 정의하고 전처리 순서를 표준화했습니다. "
                "그 결과 팀원들이 같은 기준으로 데이터를 활용할 수 있었고, 분석 결과를 더 빠르게 도출할 수 있었습니다. 이 경험을 통해 "
                "데이터 분석가는 단순히 결과를 계산하는 사람이 아니라, 해석 가능한 구조를 만드는 사람이라는 점을 배웠습니다."
            ),
            (
                "저의 강점은 복잡한 정보를 정리해 실행 가능한 형태로 바꾸는 능력입니다. ESG 관련 프로젝트를 수행하며 뉴스, 공시, 기업 정보처럼 "
                "형태가 다른 데이터를 함께 다뤄야 했습니다. 처음에는 정보가 많을수록 좋다고 생각했지만, 실제로는 연결 기준이 불분명하면 "
                "분석 품질이 떨어진다는 점을 확인했습니다. 이후 불필요한 항목을 걷어내고, 실제 활용 목적에 맞게 데이터 구조를 단순화했습니다. "
                "이 과정은 이후 모델 적용과 결과 해석의 효율을 높이는 데 도움이 됐습니다."
            ),
            (
                "저는 분석 결과를 혼자 이해하는 데서 끝내지 않고, 다른 사람이 바로 활용할 수 있도록 정리하는 편입니다. Python으로 정리한 데이터를 "
                "Tableau로 시각화해 팀원들과 공유했던 경험이 있습니다. 단순히 그래프를 만드는 데 그치지 않고, 어떤 지표를 봐야 의사결정에 도움이 되는지 "
                "함께 고민했습니다. 이를 통해 분석의 완성도는 모델 성능뿐 아니라, 결과를 전달하는 방식에서도 결정된다는 점을 배웠습니다."
            ),
        ],
    )


def main() -> None:
    client = OllamaClient()

    print("=" * 60)
    print(f"사용 모델: {client.model}")
    print("Ollama 연결 확인 중...")

    if not client.health_check():
        print("Ollama 서버에 연결할 수 없습니다.")
        print("1) Ollama 앱이 실행 중인지 확인하세요.")
        print("2) 터미널에서 'ollama list' 또는 'ollama run qwen2.5:7b'가 되는지 확인하세요.")
        return

    print("연결 성공")
    print("초안 생성 중...\n")

    sample_request = load_sample_request()

    try:
        draft = client.generate_draft(sample_request)
    except RuntimeError as exc:
        print(str(exc))
        return

    print("[생성 결과]")
    print("-" * 60)
    print(draft)
    print("-" * 60)


if __name__ == "__main__":
    main()
