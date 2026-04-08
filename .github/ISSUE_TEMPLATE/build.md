---
name: Build
about: 빌드 시스템, 의존성 패키지 설치 및 환경 설정 변경
title: "[Build] "
labels: build
assignees: ""
---

## 🏗 Build / Dependency Changes
어떤 도구나 패키지에 변화가 생겼는지 작성합니다.
- (예: `requirements.txt`에 `langchain-openai` 패키지 추가)
- (예: Python 버전을 3.9에서 3.11로 업그레이드)
- (예: Streamlit 배포 환경 설정 변경)

---

## ❓ Why is this change needed?
변경이 필요한 이유를 설명합니다.
- (예: RAG 성능 개선을 위해 최신 벡터 라이브러리 사용 필요)
- (예: 기존 라이브러리 보안 취약점 발견으로 인한 버전 업데이트)

---

## 📋 Checklist for Team
팀원들이 이 변경사항을 적용하기 위해 해야 할 일입니다.
- [ ] `pip install -r requirements.txt` 실행 필요 여부
- [ ] `.env` 파일에 새로운 환경변수 추가 필요 여부
- [ ] 로컬 캐시 삭제 및 재빌드 필요 여부

---

## ⚠️ Potential Impact
이 변경으로 인해 영향받을 수 있는 부분이나 주의사항을 작성합니다.
- (예: 특정 함수가 Deprecated 되어 코드 수정이 필요할 수 있음)
- (예: 빌드 시간이 이전보다 길어질 수 있음)

---
챠
## 📎 References
참고한 공식 문서나 관련 이슈 링크를 첨부해 주세요.
-