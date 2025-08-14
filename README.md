# Slack Chatbot

마크애니 사내 문서 기반 Q&A, 번역, 회의실 예약 기능을 제공하는 Slack 봇입니다.

## 주요 기능

### 📋 Q&A 시스템
- **멘션**: `@봇이름 질문내용`
- **슬래시 명령어**: `/질문 질문내용`
- RAG 기반 사내 문서 검색 및 답변
- 한영 자동 번역 지원

### 🌐 번역 기능
- `/영어로 텍스트` - 한국어를 영어로 번역
- `/일어로 텍스트` - 한국어를 일본어로 번역
- 확인/취소 버튼으로 채널 공유 제어
- DM 및 채널 모두 지원

### 📅 회의실 예약
- `/예약 회의실명 날짜 시간 제목` - 회의실 예약
- `/예약취소 회의실명 날짜 시간` - 예약 취소
- `/빈회의실 날짜 시간` - 사용 가능한 회의실 조회
- `/일정 @사용자 날짜 시간` - 참석자 일정 확인

### 🔧 관리 기능
- `/회수` - 문서 회수 요청 자동 처리 (권한 필요)

## 환경 설정

### 필수 환경변수
```env
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_APP_TOKEN=xapp-...
SLACK_CHANNEL_ID=...
AUTHORIZED_USERS=user1,user2
GOOGLE_API_KEY=...
FAISS_FOLDER=faiss_index
```

### 필수 파일
- `service-account-key.json` - Google Calendar API 인증
- `user_credential.json` - 웹 자동화 로그인 정보
- `faiss_index/` - 벡터 데이터베이스

## 설치 및 실행

```bash
pip install -r requirements.txt
python slack_chatbot.py
```

## 지원 회의실
- 7층 회의실1, 7층 회의실2
- 10층 1회의실, 10층 2회의실, 10층 대회의실
- 13층 1회의실, 13층 2회의실