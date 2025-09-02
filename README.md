# LottoRecommender   
로또 당첨 데이터 기반 추천 시스템 (FastAPI + React)

---

## 프로젝트 소개
과거 로또 당첨 데이터를 기반으로 **LSTM 모델**을 학습하여 번호를 추천해주는 웹 애플리케이션입니다.  

- 사용자가 원하는 세트 수를 지정하면, 모델과 규칙 기반 필터링을 통해 추천 번호를 제공합니다.  
- 번호는 색상별 공으로 시각화되어 직관적으로 확인할 수 있습니다.  

---

## 기술 스택
### Backend
- **언어**: Python 3.10  
- **프레임워크/라이브러리**: FastAPI, Uvicorn, SQLAlchemy, Pydantic, Requests  
- **데이터베이스**: SQLite  
- **API**: REST API 설계 및 구현  

### Frontend
- **언어**: JavaScript (ES6+)  
- **프레임워크/라이브러리**: React, Axios, CSS  

### AI/모델링
- **수치/데이터 처리**: NumPy, scikit-learn  
- **딥러닝 프레임워크**: PyTorch, TensorFlow/Keras  

---

## 프로젝트 구조
```
LottoRecommender/
├─ backend/ # FastAPI 서버
│ ├─ app/ # API, DB 모델, 서비스 로직
│ └─ requirements.txt
├─ frontend/ # React 앱
│ ├─ src/ # UI 컴포넌트 및 페이지
│ └─ package.json
└─ .gitignore
```
---

## ✨ 주요 기능
- **데이터 수집**: 로또 당첨 데이터 자동 크롤링 및 DB 저장  
- **추천 로직**  
  - LSTM 기반 예측 모델  
  - 규칙 기반 필터링 (중복 제거, 범위 제약 등)  
- **API 서버**  
  - 추천 번호 조회 API  
  - 과거 데이터 조회 API  
- **프론트엔드 UI**  
  - 번호 선택 그리드 (7×7)  
  - 추천 결과 시각화 및 사용자 입력 반영  

---

## 🚀 실행 방법
### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
### Frontend
```bash
cd frontend
npm install
npm start
```

---
## 성과 및 배운 점
- 백엔드와 프론트엔드를 연동하여 end-to-end 서비스를 직접 구현
- AI 모델을 실제 서비스 API로 제공하는 과정 경험
- React를 통한 사용자 친화적인 UI/UX 개선
- FastAPI + SQLAlchemy 기반 REST API 개발 및 데이터 관리 학습 경험
- PyTorch/TensorFlow를 활용한 딥러닝 모델 학습 경험
