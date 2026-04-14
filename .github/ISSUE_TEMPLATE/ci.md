---
name: Ci
about: CI/CD 파이프라인 및 자동화 설정 변경
title: "[Ci] "
labels: ci
assignees: ""
---

## 🔄 Workflow Changes
자동화 프로세스 중 변경되는 부분을 설명합니다.
- (예: GitHub Actions 워크플로우 YAML 파일 수정)
- (예: 단위 테스트 자동 실행 단계 추가)
- (예: 배포 자동화(CD) 대상 서버 또는 환경 변경)

---

## 🎯 Motivation
이 자동화 설정 변경이 필요한 이유를 설명합니다.
- (예: 코드 푸시 시마다 자동으로 코드 스타일을 점검하기 위함)
- (예: 배포 과정에서의 수동 실수를 방지하고 속도를 개선하기 위함)

---

## 🛠 Automation Details
구체적으로 어떤 작업이 수행되는지 기술합니다.
- **Trigger:** (예: main 브랜치로의 Pull Request, 특정 태그 생성 시)
- **Action:** (예: Pytest 실행, Docker 이미지 빌드, Streamlit Cloud 재배포)
- **Environment:** (예: Ubuntu-latest, Python 3.11)

---

## 📋 Checklist for Team
팀원들이 인지하거나 협조해야 할 사항을 확인합니다.
- [ ] GitHub Repository Secret(API Key 등) 설정 필요 여부
- [ ] 특정 브랜치 보호 규칙(Branch Protection Rule) 적용 여부
- [ ] CI 통과 실패 시 머지(Merge) 제한 여부

---

## ⚠️ Potential Impact
배포 과정이나 개발 흐름에 미칠 수 있는 영향을 명시합니다.
- (예: CI 검사 단계 추가로 인해 PR 머지 시간이 다소 길어질 수 있음)
- (예: 배포 환경 변수 설정이 잘못될 경우 일시적으로 서비스 접속이 불가능할 수 있음)

---

## 📎 References
참고한 워크플로우 예시, 공식 문서 또는 관련 링크를 첨부합니다.
-