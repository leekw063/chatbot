import os
import re
import csv
import json
import time
import asyncio
import logging
from queue import Queue
from threading import Thread
from typing import List, Optional, Dict
from datetime import datetime, timedelta

# Google Calendar API 관련 라이브러리
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_sdk.errors import SlackApiError
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from tenacity import retry, stop_after_attempt, wait_exponential
from decouple import config
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    "%Y-%m-%d %H:%M:%S"
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler(
    "slack_bot.log",
    mode="a",
    encoding="utf-8"
)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class SlackBot:
    def __init__(self) -> None:
        load_dotenv()
        self.SLACK_BOT_TOKEN = config("SLACK_BOT_TOKEN")
        self.SLACK_SIGNING_SECRET = config("SLACK_SIGNING_SECRET")
        self.SLACK_APP_TOKEN = config("SLACK_APP_TOKEN")
        self.SLACK_CHANNEL_ID = config("SLACK_CHANNEL_ID")
        self.AUTHORIZED_USERS = config("AUTHORIZED_USERS", cast=lambda v: [s.strip() for s in v.split(',')])
        self.FAISS_FOLDER = config("FAISS_FOLDER", default="faiss_index")
        self.CSV_LOG_PATH = config("CSV_LOG_PATH", default="qa_logs.csv")
        self.CREDENTIAL_FILE = config("CREDENTIAL_FILE", default="user_credential.json")
        self.TIMESTAMP_FILE = config("TIMESTAMP_FILE", default="last_timestamp.txt")
        self.KEYWORDS = ["취소", "삭제", "회수", "반려", "(주)마크애니"]
        self.GOOGLE_API_KEY = config("GOOGLE_API_KEY")

        # --- Google Calendar 관련 설정 ---
        self.GOOGLE_CREDS_FILE = 'service-account-key.json'
        self.GOOGLE_TOKEN_FILE = 'token.json'
        self.CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']
        self.CALENDAR_RESOURCE_IDS = {
            "7층 회의실1": "markany.com_3537333337393130353833@resource.calendar.google.com",
            "7층 회의실2": "markany.com_3533333936353531333336@resource.calendar.google.com",
            "10층 1회의실": "c_188djttn33qf2jjpiohcdc45dmeuo@resource.calendar.google.com",
            "10층 2회의실": "c_1884bh9dmmseoht5jihl0em1f1ohg@resource.calendar.google.com",
            "10층 대회의실": "markany.com_34313638323834343330@resource.calendar.google.com",
            "13층 1회의실": "c_1885t7k2cn73gjoailu9ppdrfb126@resource.calendar.google.com",
            "13층 2회의실": "c_1888janh7sbvghatjg96v4ha811go@resource.calendar.google.com"
        }

        if not all([self.SLACK_BOT_TOKEN, self.SLACK_SIGNING_SECRET, self.SLACK_APP_TOKEN, self.GOOGLE_API_KEY]):
            raise ValueError("필수 환경변수가 누락되었습니다.")

        self.app = AsyncApp(token=self.SLACK_BOT_TOKEN, signing_secret=self.SLACK_SIGNING_SECRET)
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.llm = None
        self.log_queue: "Queue[List[str]]" = Queue()
        self.socket_handler: Optional[AsyncSocketModeHandler] = None
        self.PROMPT_KO = PromptTemplate(template=self._ko_prompt(), input_variables=["context", "question"])

        Thread(target=self._log_writer, daemon=True).start()
        self._initialize_components()

    def register_handlers(self) -> None:
        """모든 이벤트 및 커맨드 핸들러를 등록합니다."""
        self.app.event("app_mention")(self._handle_app_mention)
        self.app.command("/질문")(self._handle_slash_question)
        self.app.command("/영어로")(self._handle_slash_translate_en)
        self.app.command("/일어로")(self._handle_slash_translate_jp)
        self.app.command("/회수")(self._handle_collect_command)
        self.app.command("/예약")(self._handle_booking_command)
        self.app.command("/예약취소")(self._handle_cancellation_command)
        self.app.command("/빈회의실")(self._handle_find_room_command)
        self.app.command("/일정")(self._handle_schedule_check_command)
        self.app.action("feedback_helpful")(self._handle_feedback)
        self.app.action("feedback_unhelpful")(self._handle_feedback)

        # 예약 및 취소 관련 버튼 핸들러
        self.app.action("confirm_cancel")(self._handle_confirm_cancel)
        self.app.action("deny_cancel")(self._handle_deny_cancel)
        self.app.action("confirm_book")(self._handle_confirm_book)
        self.app.action("deny_book")(self._handle_deny_book)
        self.app.action("confirm_book_anyway")(self._handle_confirm_book_anyway)
        
        # 번역 관련 버튼 핸들러
        self.app.action("confirm_translate")(self._handle_confirm_translate)
        self.app.action("cancel_translate")(self._handle_cancel_translate)

    def _initialize_components(self) -> None:
        """RAG 파이프라인과 스케줄러를 초기화합니다."""
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        if not os.path.isdir(self.FAISS_FOLDER): raise FileNotFoundError(f"FAISS index 폴더가 없습니다: {self.FAISS_FOLDER}")
        self.vectorstore = FAISS.load_local(self.FAISS_FOLDER, embeddings, allow_dangerous_deserialization=True)
        logging.info("FAISS index loaded.")
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=self.GOOGLE_API_KEY)
        logging.info("LLM & Retriever ready.")
        self.scheduler = AsyncIOScheduler(timezone='Asia/Seoul')
        self.scheduler.add_job(self._restart_socket, 'interval', hours=3)
        self.scheduler.add_job(self._cleanup_declined_events, 'interval', hours=1)

    @staticmethod
    def _ko_prompt() -> str:
        return (
            "당신은 마크애니 직원의 질문에 답변하는 친절하고 유능한 AI 챗봇입니다.\n"
            "마크애니 사내 문서만 참고하여 답변하세요.\n\n"
            "● **답변 규칙**\n"
            "1) 800자 이내 **또는** 최대 10줄 글머리표(▷) 요약\n"
            "2) 문서에 정보가 없으면  ➜  \"죄송합니다. 해당 내용은 문서에 없습니다.\" 출력\n"
            "3) 정중하고 정리된 문장으로 서술형으로 답변.\n\n"
            "**문서:**\n{context}\n\n"
            "**[😃Question]** {question}\n\n"
            "**[🧐Answer]**"
        )

    @staticmethod
    def _clean(text: str) -> str:
        cleaned = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        return re.sub(r"(\*\*|\*)", "", cleaned).strip()

    async def _translate(self, text: str, target_lang: str) -> str:
        try:
            prompt = (f"Translate the following text to {target_lang}.\nOnly provide the translated text without any extra commentary.\n\n{text}")
            resp = await self.llm.ainvoke(prompt)
            return resp.content.strip()
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return text  # 번역 실패 시 원문 반환

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=6))
    async def _generate_response(self, question: str) -> str:
        try:
            lang = detect(question)
        except LangDetectException:
            lang = "ko"
        if lang.startswith("en"):
            ko_question = await self._translate(question, "Korean")
            docs = await self.retriever.ainvoke(ko_question)
            context = "\n".join(d.page_content for d in docs) if docs else ""
            ko_prompt = self.PROMPT_KO.format(context=context, question=ko_question)
            ko_answer = (await self.llm.ainvoke(ko_prompt)).content.strip()
            return await self._translate(ko_answer, "English")
        
        docs = await self.retriever.ainvoke(question)
        context = "\n".join(d.page_content for d in docs) if docs else ""
        prompt = self.PROMPT_KO.format(context=context, question=question)
        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    async def _handle_app_mention(self, body, say, client):
        user_question = re.sub(r"<@\w+>", "", body["event"]["text"]).strip()
        user_id = body["event"]["user"]
        ts = body["event"]["ts"]
        channel_id = body["event"]["channel"]
        
        try:
            thinking_msg = await say(text=f"🤔 '{user_question}'에 대해 답변을 준비 중입니다...", thread_ts=ts)
            answer = await self._generate_response(user_question)
            await client.chat_update(channel=channel_id, ts=thinking_msg["ts"], text=f"[😃Question] {user_question}\n[🧐Answer]\n{self._clean(answer)}")
            self.log_queue.put([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, user_question, answer, "helpful"])
        except Exception:
            logging.exception("Error responding to mention")
            await client.chat_postMessage(channel=channel_id, thread_ts=ts, text="❌ 답변 생성 중 오류가 발생했습니다.")

    async def _handle_slash_question(self, ack, respond, command):
        await ack()
        asyncio.create_task(self._process_question(command, respond))

    async def _process_question(self, command, respond):
        user_question = command.get("text", "").strip()
        user_id = command.get("user_id", "")
        try:
            await respond(f"🤔 '{user_question}'에 대해 답변을 준비 중입니다...")
            answer = await self._generate_response(user_question)
            await respond(f"[😃Question] {user_question}\n[🧐Answer]\n{self._clean(answer)}")
            self.log_queue.put([datetime.utcnow().isoformat(), user_id, user_question, answer, ""])
        except Exception:
            logging.exception("Error in /질문 command")
            await respond("❌ 답변 생성 중 오류가 발생했습니다.")

    async def _handle_translate_common(self, ack, body, say, client, target_lang: str, lang_display: str):
        """공통 번역 처리 로직"""
        await ack()
        user_id = body["user_id"]
        text = body.get("text", "").strip()
        channel_id = body.get("channel_id")
        
        if not text:
            try:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text=f"번역할 텍스트를 입력해주세요. 예: `/{lang_display}로 안녕하세요`"
                )
            except:
                await say(f"번역할 텍스트를 입력해주세요. 예: `/{lang_display}로 안녕하세요`")
            return
            
        try:
            translated = await self._translate(text, target_lang)
            user_info = await client.users_info(user=user_id)
            user_name = user_info["user"]["real_name"] or user_info["user"]["display_name"] or user_info["user"]["name"]
            
            # 채널 타입 확인
            try:
                channel_info = await client.conversations_info(channel=channel_id)
                is_dm = channel_info["channel"]["is_im"]
                share_text = f"DM에 {lang_display} 번역 결과를 표시하시겠습니까?" if is_dm else f"채팅창에 {lang_display} 번역 결과를 공유하시겠습니까?"
            except:
                is_dm = False
                share_text = f"{lang_display} 번역 결과를 공유하시겠습니까?"
            
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"**원문:** {text}\n**{lang_display} 번역:** {translated}\n\n{share_text}"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "확인"},
                            "style": "primary",
                            "action_id": "confirm_translate",
                            "value": json.dumps({
                                "user_name": user_name,
                                "translated_text": translated,
                                "original_text": text,
                                "target_lang": lang_display
                            })
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "취소"},
                            "action_id": "cancel_translate"
                        }
                    ]
                }
            ]
            
            try:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    blocks=blocks,
                    text="번역 확인"
                )
            except:
                # DM에서는 ephemeral 메시지가 작동하지 않으므로 일반 메시지로 전송
                await say(blocks=blocks)
            
        except Exception as e:
            logging.error(f"Translation error: {e}")
            try:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="번역 중 오류가 발생했습니다."
                )
            except:
                await say("번역 중 오류가 발생했습니다.")

    async def _handle_slash_translate_en(self, ack, body, say, client):
        await self._handle_translate_common(ack, body, say, client, "English", "영어")

    async def _handle_slash_translate_jp(self, ack, body, say, client):
        await self._handle_translate_common(ack, body, say, client, "Japanese", "일본어")

    async def _handle_confirm_translate(self, ack, body, say, client):
        """번역 확인 버튼 처리"""
        await ack()
        
        try:
            value_data = json.loads(body["actions"][0]["value"])
            user_name = value_data["user_name"]
            translated_text = value_data["translated_text"]
            target_lang = value_data["target_lang"]
            channel_id = body["channel"]["id"]
            
            # 채널 정보 확인 (DM인지 일반 채널인지)
            try:
                channel_info = await client.conversations_info(channel=channel_id)
                is_dm = channel_info["channel"]["is_im"]
            except:
                is_dm = False
            
            if is_dm:
                # DM인 경우 chat_postMessage 직접 사용
                await client.chat_postMessage(
                    channel=channel_id,
                    text=f"[{user_name}]: {translated_text}"
                )
                success_msg = f"✅ {target_lang} 번역이 DM에 표시되었습니다."
            else:
                # 일반 채널인 경우 기존 방식 사용
                await say(
                    text=f"[{user_name}]: {translated_text}",
                    response_type="in_channel"
                )
                success_msg = f"✅ {target_lang} 번역이 채팅창에 공유되었습니다."
            
            # 원본 메시지 업데이트 (버튼 제거)
            await client.chat_update(
                channel=channel_id,
                ts=body["message"]["ts"],
                text=success_msg
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"JSON parsing error in confirm_translate: {e}")
            try:
                await client.chat_postMessage(
                    channel=body["channel"]["id"],
                    text="번역 공유 중 오류가 발생했습니다."
                )
            except:
                await say("번역 공유 중 오류가 발생했습니다.")
        except Exception as e:
            logging.error(f"Confirm translate error: {e}")
            try:
                await client.chat_postMessage(
                    channel=body["channel"]["id"],
                    text="번역 공유 중 오류가 발생했습니다."
                )
            except:
                await say("번역 공유 중 오류가 발생했습니다.")

    async def _handle_cancel_translate(self, ack, body, client):
        """번역 취소 버튼 처리"""
        await ack()
        
        try:
            # 원본 메시지 업데이트
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=body["message"]["ts"],
                text="❌ 번역이 취소되었습니다."
            )
        except Exception as e:
            logging.error(f"Cancel translate error: {e}")
            try:
                await client.chat_postMessage(
                    channel=body["channel"]["id"],
                    text="❌ 번역이 취소되었습니다."
                )
            except:
                pass

    async def _handle_feedback(self, ack, body, respond):
        await ack()
        fb_id = body["actions"][0]["action_id"]
        await respond(f"피드백 감사합니다! ({fb_id})")

    def _log_writer(self) -> None:
        while True:
            batch: List[List[str]] = []
            while len(batch) < 100 and not self.log_queue.empty():
                batch.append(self.log_queue.get())
            if batch:
                try:
                    new_file = not os.path.isfile(self.CSV_LOG_PATH)
                    with open(self.CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if new_file:
                            writer.writerow(["timestamp", "user", "question", "answer", "sources"])
                        writer.writerows(batch)
                except Exception as e:
                    logging.error(f"CSV 기록 오류: {e}")
            else:
                time.sleep(10)  # 배치가 비어있을 때만 대기
            time.sleep(1)  # CPU 사용량 감소

    def _read_last_timestamp(self):
        if os.path.exists(self.TIMESTAMP_FILE):
            with open(self.TIMESTAMP_FILE, 'r') as f:
                return f.read().strip()
        return (datetime.now() - timedelta(days=1)).timestamp()

    def _save_last_timestamp(self, ts):
        with open(self.TIMESTAMP_FILE, 'w') as f:
            f.write(str(ts))

    async def _fetch_new_requests(self, start_ts):
        requests_to_process, latest_message_ts = [], start_ts
        try:
            response = await self.app.client.conversations_history(channel=self.SLACK_CHANNEL_ID, oldest=str(start_ts))
            messages = response.get("messages", [])
            if not messages:
                return [], None
            latest_message_ts = messages[0].get("ts", latest_message_ts)
            
            for msg in messages:
                text = msg.get("text", "")
                user_id = msg.get("user")
                if any(keyword in text for keyword in self.KEYWORDS) and user_id:
                    doc_match = re.search(r'\(주\)마크애니-(\d+)', text)
                    user_name = None
                    try:
                        user_info = await self.app.client.users_profile_get(user=user_id)
                        raw_user_name = user_info.get("profile", {}).get("real_name")
                        if raw_user_name:
                            korean_name_match = re.search(r'[가-힣]+', raw_user_name)
                            if korean_name_match:
                                user_name = korean_name_match.group(0)
                            else:
                                user_name = raw_user_name
                    except SlackApiError as e:
                        logging.error(f"Slack API Error (users_profile_get) for {user_id}: {e}")
                    
                    if user_name and doc_match:
                        requests_to_process.append({"name": user_name.strip(), "doc_number": doc_match.group(1)})
            return requests_to_process, latest_message_ts
        except SlackApiError as e:
            logging.error(f"Slack API 오류: {e.response['error']}")
            return [], None

    def _get_credentials(self):
        if not os.path.exists(self.CREDENTIAL_FILE):
            logging.error(f"오류: {self.CREDENTIAL_FILE} 파일이 없습니다.")
            return None, None
        try:
            with open(self.CREDENTIAL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["id"], data["pw"]
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Credential file parsing error: {e}")
            return None, None

    def _process_single_cancellation(self, driver, request_data):
        employee_name, doc_number = request_data['name'], request_data['doc_number']
        try:
            wait = WebDriverWait(driver, 15)
            logging.info(f"▶ {employee_name} / {doc_number} 건 처리 시작...")
            main_menu = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="topMenu700000000"]')))
            driver.execute_script("arguments[0].click();", main_menu)
            sub_menu = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="701050300_anchor"]')))
            driver.execute_script("arguments[0].click();", sub_menu)
            logging.info("▶ iFrame으로 전환합니다...")
            wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "_content")))
            all_menu_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="all_menu_btn"]')))
            driver.execute_script("arguments[0].click();", all_menu_button)
            emp_name_input = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="emp_name"]')))
            emp_name_input.clear()
            emp_name_input.send_keys(employee_name)
            search_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="searchButton"]')))
            driver.execute_script("arguments[0].click();", search_button)
            logging.info("▶ 검색을 실행했습니다.")
            time.sleep(2)
            result_rows = driver.find_elements(By.XPATH, '//*[@id="grid_1"]//tbody/tr')
            target_found = False
            if not result_rows or ("데이터가 없습니다" in result_rows[0].text):
                return f"❌ {employee_name} / {doc_number} 인사팀 확인 필요 (검색 결과 없음)"
            for row in result_rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) > 14 and cells[14].text.endswith(doc_number):
                    checkbox = row.find_element(By.XPATH, './/td[1]//input')
                    if checkbox.is_enabled():
                        driver.execute_script("arguments[0].click();", checkbox)
                        target_found = True
                        logging.info("▶ 체크박스를 선택했습니다.")
                    else:
                        return f"❌ {employee_name} / {doc_number} 인사팀 확인 필요 (이미 처리된 항목)"
                    break
            if not target_found:
                return f"❌ {employee_name} / {doc_number} 인사팀 확인 필요 (문서번호 불일치)"
            time.sleep(1)
            delete_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="deleteResetBtn"]')))
            driver.execute_script("arguments[0].click();", delete_button)
            time.sleep(1)
            reason_input = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="reason"]')))
            reason_input.send_keys("개인요청")
            save_button_xpath = "//input[@value='저장']"
            save_button = wait.until(EC.element_to_be_clickable((By.XPATH, save_button_xpath)))
            driver.execute_script("arguments[0].click();", save_button)
            for i in range(3):
                logging.info(f"▶ {i+1}번째 확인창을 처리합니다...")
                wait.until(EC.alert_is_present())
                alert = driver.switch_to.alert
                logging.info(f"  > 확인창 내용: {alert.text}")
                alert.accept()
                time.sleep(1)
            return f"✅ {employee_name} / {doc_number} 회수/삭제 완료"
        except Exception as e:
            logging.error(f"처리 중 오류: {str(e)}")
            return f"❌ {employee_name} / {doc_number} 인사팀 확인 필요 (자동화 오류)"

    async def _run_cancellation_task(self, say, user_id, channel_id):
        logging.info("--- 백그라운드 회수 작업 시작 ---")
        last_ts = self._read_last_timestamp()
        requests, new_ts = await self._fetch_new_requests(last_ts)
        
        success_report = []
        failure_report = []

        if not requests:
            failure_report.append("처리할 신규 요청이 없습니다.")
        else:
            USER_ID, USER_PASSWORD = self._get_credentials()
            if not USER_ID:
                failure_report.append("오류: 로그인 정보 파일(user_credential.json)을 찾을 수 없습니다.")
            else:
                driver = None
                try:
                    options = webdriver.ChromeOptions()
                    options.add_argument("--headless")
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-dev-shm-usage")
                    driver = webdriver.Chrome(options=options)
                    
                    wait = WebDriverWait(driver, 15)
                    driver.get("https://ngw.markany.com/gw/uat/uia/egovLoginUsr.do")
                    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="userId"]'))).send_keys(USER_ID)
                    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="userPw"]'))).send_keys(USER_PASSWORD)
                    login_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="login_b2_type"]/div[2]/div/form/fieldset/div[2]/div')))
                    driver.execute_script("arguments[0].click();", login_button)
                    logging.info(f"▶ {USER_ID} 계정으로 로그인합니다.")
                    for req in requests:
                        result = await asyncio.to_thread(self._process_single_cancellation, driver, req)
                        if result.startswith("✅"):
                            success_report.append(result)
                        else:
                            failure_report.append(result)
                        await asyncio.sleep(1)
                finally:
                    if driver:
                        driver.quit()
                        logging.info("\n--- Selenium 작업 완료. 브라우저가 종료되었습니다. ---")
        if new_ts:
            self._save_last_timestamp(new_ts)

        if success_report:
            report_text = f"--- `/회수` 작업 성공 내역 ---\n" + "\n".join(success_report)
            logging.info(report_text)
            await say(text=report_text)
        
        if failure_report:
            report_text = f"--- `/회수` 작업 실패/오류 내역 ---\n" + "\n".join(failure_report)
            logging.warning(report_text)
            try:
                await self.app.client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text=report_text
                )
            except SlackApiError as e:
                logging.error(f"Could not send ephemeral message: {e}")

    async def _handle_collect_command(self, ack, say, command):
        user_id = command["user_id"]
        channel_id = command["channel_id"]
        if user_id not in self.AUTHORIZED_USERS:
            await ack("⚠️ 권한이 없습니다.")
            return
        await ack("✅ `/회수` 요청을 접수했습니다. 잠시 후 결과가 처리됩니다...")
        asyncio.create_task(self._run_cancellation_task(say, user_id, channel_id))

    def _get_calendar_service(self):
        creds = None
        if os.path.exists(self.GOOGLE_TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(self.GOOGLE_TOKEN_FILE, self.CALENDAR_SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.GOOGLE_CREDS_FILE, self.CALENDAR_SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.GOOGLE_TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        return build('calendar', 'v3', credentials=creds)

    async def _parse_booking_request(self, user_text: str, is_cancellation: bool = False):
        action = "취소할" if is_cancellation else "예약할"
        prompt_action_verb = "추출" if user_text else "생성"
        # /일정 커맨드 처리를 위해 동사 수정
        user_facing_action = "조회할" if "일정" in action else "찾을" if action == "예약할" and "찾" in user_text else action

        resource_key_text = f"'resource'는 반드시 다음 중 하나여야 해: {list(self.CALENDAR_RESOURCE_IDS.keys())}." if "회의실" in user_text else ""
        
        prompt = (
            f"다음 사용자 요청에서 '{user_facing_action}' 정보를 {prompt_action_verb}해서 ISO 8601 형식의 JSON으로 반환해줘.\n"
            f"JSON의 키는 반드시 'resource', 'title', 'start_time', 'end_time'을 사용해야 해. {resource_key_text}\n"
            f"시간 값은 반드시 한국 표준시(KST) 기준이며, +09:00 형식의 타임존 오프셋을 포함해야 해. (예: 2025-08-13T13:00:00+09:00)\n"
            f"현재 시간은 {datetime.now().astimezone().isoformat()} 이야. '내일', '모레', '오후' 같은 상대적인 시간 표현을 이 시간을 기준으로 계산해야 해.\n"
            f"'시작 시간'이 입력되지 않으면 현재 시간으로 인식해. 일자가 입력되지 않으면 오늘일자를 기준. 월(month)가 입력되지 않으면 오늘이 속하는 월을 기준으로 해. \n"
            f"회의 제목이 없으면 '회의'라고 해줘. 회의 시간은 기본 1시간이야.\n"
            f"시간 해석 규칙: '13시'는 '13:00'을 의미해. '오후 1시'도 '13:00'이야. '오전/오후' 언급이 없이 n시라고 한다면 08시~19시 사이에 있는 오전/오후가 입력된 시간으로 해석해줘.(eg. 1시는 오후 1시(13시) \n"
            f"사용자 요청: '{user_text}'\n\n"
            "JSON:"
        )
        response = await self.llm.ainvoke(prompt)
        try:
            json_str_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            if not json_str_match:
                json_str = response.content
            else:
                json_str = json_str_match.group(1)

            parsed_data = json.loads(json_str)
            logging.info(f"Parsed booking request: {parsed_data}")
            return parsed_data
        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Failed to parse JSON from LLM response: {response.content}, Error: {e}")
            return None

    async def _check_availability(self, parsed_data: dict, attendee_emails_map: dict) -> dict:
        try:
            service = self._get_calendar_service()
            resource_name = parsed_data.get("resource")
            resource_id = self.CALENDAR_RESOURCE_IDS.get(resource_name)
            
            if not resource_id:
                return {"error": f"❌ '{resource_name}'을(를) 찾을 수 없습니다. 회의실 이름을 확인해주세요."}

            start_time = parsed_data.get("start_time")
            end_time = parsed_data.get("end_time")

            all_calendars_to_check = [{"id": resource_id}] + [{"id": email} for email in attendee_emails_map.keys()]

            freebusy_body = {"timeMin": start_time, "timeMax": end_time, "items": all_calendars_to_check}
            freebusy_result = service.freebusy().query(body=freebusy_body).execute()
            
            is_room_busy = bool(freebusy_result['calendars'][resource_id]['busy'])
            
            busy_attendees = []
            for email, name in attendee_emails_map.items():
                if freebusy_result['calendars'].get(email, {}).get('busy'):
                    busy_attendees.append(name)
            
            return {
                "is_room_busy": is_room_busy,
                "busy_attendees": busy_attendees,
                "error": None
            }
        except HttpError as e:
            logging.error(f"Google Calendar API error: {e}")
            return {"error": f"❌ Google Calendar API 오류: {e}"}
        except Exception as e:
            logging.error(f"Availability check error: {e}")
            return {"error": "❌ 일정 확인 중 오류가 발생했습니다."}

    def _create_calendar_event(self, parsed_data: dict, requester_email: str, attendee_emails_map: dict):
        try:
            service = self._get_calendar_service()
            resource_name = parsed_data.get("resource")
            resource_id = self.CALENDAR_RESOURCE_IDS.get(resource_name)
            
            attendees_list = [{'email': resource_id}, {'email': requester_email}]
            for email in attendee_emails_map.keys():
                attendees_list.append({'email': email})

            event = {
                'summary': parsed_data.get("title", "회의"),
                'start': {'dateTime': parsed_data.get("start_time"), 'timeZone': 'Asia/Seoul'},
                'end': {'dateTime': parsed_data.get("end_time"), 'timeZone': 'Asia/Seoul'},
                'attendees': attendees_list,
                'reminders': {'useDefault': True},
            }
            created_event = service.events().insert(calendarId='primary', body=event, sendUpdates='all').execute()
            logging.info(f"Event created: {created_event.get('htmlLink')}")

            start_dt = datetime.fromisoformat(parsed_data.get("start_time"))
            end_dt = datetime.fromisoformat(parsed_data.get("end_time"))
            start_dt_str = start_dt.strftime('%#m월 %#d일 %p %#I:%M').replace('AM', '오전').replace('PM', '오후')
            end_dt_str = end_dt.strftime('%p %#I:%M').replace('AM', '오전').replace('PM', '오후')
            
            return f"✅ [{resource_name}] 예약 완료: {parsed_data.get('title')}\n({start_dt_str} ~ {end_dt_str})"
        except HttpError as error:
            logging.error(f'Google Calendar API error: {error}')
            return f"❌ Google Calendar API 오류: {error}"
        except Exception as e:
            logging.error(f"이벤트 생성 중 오류: {e}", exc_info=True)
            return "❌ 이벤트 생성 중 알 수 없는 오류 발생"
            
    async def _lookup_user_by_name(self, name):
        try:
            response = await self.app.client.users_list()
            for user in response["members"]:
                profile = user.get("profile", {})
                real_name = profile.get("real_name", "").lower()
                display_name = profile.get("display_name", "").lower()
                user_name = user.get("name", "").lower()
                
                if name.lower() in [real_name, display_name, user_name]:
                    email = profile.get("email")
                    if email:
                        return user["id"], email, profile.get("real_name_normalized", name)
            return None, None, None
        except SlackApiError as e:
            logging.error(f"Error looking up user by name {name}: {e.response['error']}")
            return None, None, None

    async def _process_booking(self, command, respond, client):
        user_text = command.get("text", "").strip()
        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id")
        
        mentioned_user_ids = re.findall(r'<@(\w+)>', user_text)
        non_standard_mentions = [match.strip() for match in re.findall(r'@([\w.-]+)', user_text)]
        
        cleaned_text = re.sub(r'<@\w+\s*>', '', user_text)
        cleaned_text = re.sub(r'@[\w.-]+\s*', '', cleaned_text).strip()
        
        try:
            user_info = await self.app.client.users_profile_get(user=user_id)
            requester_email = user_info["profile"].get("email")
            if not requester_email:
                await respond(text="❌ 요청자의 이메일을 가져올 수 없습니다. Slack 프로필을 확인해주세요.")
                return
            
            parsed_data = await self._parse_booking_request(cleaned_text)
            if not parsed_data or not parsed_data.get('resource'):
                await respond(text="❌ 요청을 이해하지 못했습니다. 날짜, 시간, 회의실 이름을 포함하여 다시 요청해주세요.")
                return

            attendee_emails_map = {}
            all_attendee_names = []
            
            if mentioned_user_ids:
                for mentioned_id in mentioned_user_ids:
                    try:
                        mentioned_user_info = await self.app.client.users_profile_get(user=mentioned_id)
                        email = mentioned_user_info.get("profile", {}).get("email")
                        name = mentioned_user_info.get("profile", {}).get("real_name_normalized", f"@{mentioned_id}")
                        if email and email not in attendee_emails_map:
                            attendee_emails_map[email] = name
                            all_attendee_names.append(name)
                    except SlackApiError as e:
                        logging.warning(f"Failed to get email for user {mentioned_id}: {e.response['error']}")

            if non_standard_mentions:
                for name in non_standard_mentions:
                    _, email, real_name = await self._lookup_user_by_name(name)
                    if email and email not in attendee_emails_map:
                        attendee_emails_map[email] = real_name
                        all_attendee_names.append(real_name)
            
            start_dt = datetime.fromisoformat(parsed_data['start_time'])
            end_dt = datetime.fromisoformat(parsed_data['end_time'])
            start_str = start_dt.strftime('%#m월 %#d일 %p %#I:%M').replace('AM', '오전').replace('PM', '오후')
            end_str = end_dt.strftime('%p %#I:%M').replace('AM', '오전').replace('PM', '오후')
            attendees_str = ", ".join(all_attendee_names) if all_attendee_names else "없음"

            confirmation_text = (
                f"다음 내용으로 예약을 진행할까요?\n"
                f"*회의실:* {parsed_data.get('resource')}\n"
                f"*제목:* {parsed_data.get('title')}\n"
                f"*시간:* {start_str} ~ {end_str}\n"
                f"*참석자:* {attendees_str}"
            )
            
            action_value = {
                "parsed_data": parsed_data,
                "requester_email": requester_email,
                "attendee_emails_map": attendee_emails_map
            }
            
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": confirmation_text}},
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "확인"},
                            "style": "primary",
                            "value": json.dumps(action_value),
                            "action_id": "confirm_book"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "취소"},
                            "action_id": "deny_book"
                        }
                    ]
                }
            ]
            
            await client.chat_postEphemeral(channel=channel_id, user=user_id, blocks=blocks, text="예약 확인")

        except Exception as e:
            logging.error(f"Error in /예약 command: {e}", exc_info=True)
            await respond(text="❌ 예약 처리 중 오류가 발생했습니다.")

    async def _handle_booking_command(self, ack, respond, command, client):
        await ack()
        asyncio.create_task(self._process_booking(command, respond, client))
    
    def _find_calendar_event(self, parsed_data: dict):
        try:
            service = self._get_calendar_service()
            resource_name = parsed_data.get("resource")
            resource_id = self.CALENDAR_RESOURCE_IDS.get(resource_name)
            if not resource_id:
                return None, f"❌'{resource_name}'을(를) 찾을 수 없습니다. 회의실 이름을 확인해주세요."

            start_time_str = parsed_data.get("start_time")
            start_time = datetime.fromisoformat(start_time_str)
            time_min = (start_time - timedelta(minutes=1)).isoformat()
            time_max = (start_time + timedelta(minutes=1)).isoformat()

            events_result = service.events().list(calendarId='primary', timeMin=time_min, timeMax=time_max, singleEvents=True, orderBy='startTime').execute()
            events = events_result.get('items', [])
            
            if not events:
                return None, f"❌ 해당 시간에 '{resource_name}'으로 예약된 일정을 찾을 수 없습니다."

            event_to_delete = None
            for event in events:
                attendees = event.get('attendees', [])
                has_resource = any(a.get('email') == resource_id for a in attendees)
                if has_resource:
                    event_to_delete = event
                    break
            
            if not event_to_delete:
                return None, f"❌ 해당 시간에 '{resource_name}'으로 예약된 일정을 찾을 수 없습니다."

            return event_to_delete, None
        except HttpError as error:
            logging.error(f'An error occurred: {error}')
            return None, f"❌ Google Calendar API 오류: {error}"
        except Exception as e:
            logging.error(f"예약 취소 중 오류: {e}")
            return None, "❌ 예약 취소 중 알 수 없는 오류 발생"

    def _cancel_calendar_event(self, event_to_delete: dict):
        try:
            service = self._get_calendar_service()
            service.events().delete(calendarId='primary', eventId=event_to_delete['id'], sendUpdates='all').execute()
            
            start_time_str = event_to_delete['start'].get('dateTime', event_to_delete['start'].get('date'))
            start_dt = datetime.fromisoformat(start_time_str)
            start_dt_str = start_dt.strftime('%#m월 %#d일 %p %#I:%M').replace('AM', '오전').replace('PM', '오후')
            return f"🗑️ 예약이 취소되었습니다: {event_to_delete.get('summary', '')}\n({start_dt_str})"
        except HttpError as error:
            logging.error(f'An error occurred: {error}')
            return f"❌ Google Calendar API 오류: {error}"
        except Exception as e:
            logging.error(f"예약 취소 중 오류: {e}")
            return "❌ 예약 취소 중 알 수 없는 오류 발생"

    async def _process_cancellation(self, command, respond, client):
        user_text = command.get("text", "").strip()
        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id", "")
        try:
            cleaned_text = re.sub(r'<@\w+\s*>', '', user_text)
            cleaned_text = re.sub(r'@\w+\s*', '', cleaned_text).strip()
            parsed_data = await self._parse_booking_request(cleaned_text, is_cancellation=True)
            
            if not parsed_data or not parsed_data.get('resource'):
                await respond(text="❌ 요청을 이해하지 못했습니다. 날짜, 시간, 회의실 이름을 포함하여 다시 요청해주세요.")
                return
            
            event_to_delete, error_msg = await asyncio.to_thread(self._find_calendar_event, parsed_data)
            if error_msg:
                await respond(text=error_msg)
                return

            start_dt = datetime.fromisoformat(event_to_delete['start']['dateTime'])
            end_dt = datetime.fromisoformat(event_to_delete['end']['dateTime'])
            start_str = start_dt.strftime('%#m월 %#d일 %p %#I:%M').replace('AM', '오전').replace('PM', '오후')
            end_str = end_dt.strftime('%p %#I:%M').replace('AM', '오전').replace('PM', '오후')
            
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"다음 예약을 취소하시겠습니까?\n*회의실:* {parsed_data.get('resource')}\n*제목:* {event_to_delete.get('summary', '')}\n*시간:* {start_str} ~ {end_str}"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "확인"},
                            "style": "danger",
                            "value": json.dumps({"event_id": event_to_delete['id']}),
                            "action_id": "confirm_cancel"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "취소"},
                            "action_id": "deny_cancel"
                        }
                    ]
                }
            ]
            await client.chat_postEphemeral(channel=channel_id, user=user_id, blocks=blocks, text="예약 취소 확인 요청")
        except Exception as e:
            logging.error(f"Error in /예약취소 command: {e}", exc_info=True)
            await respond(text="❌ 예약 취소 처리 중 오류가 발생했습니다.")

    async def _handle_cancellation_command(self, ack, respond, command, client):
        await ack()
        asyncio.create_task(self._process_cancellation(command, respond, client))
    
    async def _handle_confirm_cancel(self, ack, body, client, respond):
        await ack()
        try:
            value = json.loads(body["actions"][0]["value"])
            event_id = value["event_id"]
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"JSON parsing error in confirm_cancel: {e}")
            await respond(text="❌ 요청 처리 중 오류가 발생했습니다.", replace_original=True)
            return
        
        await respond(text="예약 취소를 진행합니다...", replace_original=True)

        try:
            service = self._get_calendar_service()
            event_to_delete = await asyncio.to_thread(service.events().get, calendarId='primary', eventId=event_id)
            event_to_delete = event_to_delete.execute()
            result = await asyncio.to_thread(self._cancel_calendar_event, event_to_delete)
            await respond(text=result, replace_original=True)
        except Exception as e:
            logging.error(f"Error confirming cancellation: {e}", exc_info=True)
            await respond(text="❌ 취소 처리 중 오류가 발생했습니다.", replace_original=True)

    async def _handle_deny_cancel(self, ack, body, respond):
        await ack()
        await respond(text="취소 요청이 중단되었습니다.", replace_original=True)

    async def _handle_confirm_book(self, ack, body, client, respond):
        await ack()
        try:
            action_value = json.loads(body["actions"][0]["value"])
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"JSON parsing error in confirm_book: {e}")
            await respond(text="❌ 요청 처리 중 오류가 발생했습니다.", replace_original=True)
            return
        
        await respond(text="회의실 및 참석자 일정을 확인 중입니다...", replace_original=True)
        
        try:
            parsed_data = action_value["parsed_data"]
            attendee_emails_map = action_value["attendee_emails_map"]

            availability = await self._check_availability(parsed_data, attendee_emails_map)
            
            if availability["error"]:
                await respond(text=availability["error"], replace_original=True)
                return

            if availability["is_room_busy"]:
                resource_name = parsed_data.get("resource")
                await respond(text=f"❌ 죄송합니다. 해당 시간에는 '{resource_name}'을(를) 사용할 수 없습니다.", replace_original=True)
                return

            if availability["busy_attendees"]:
                busy_names = ", ".join(availability["busy_attendees"])
                confirmation_text = f"⚠️ {busy_names} 님은 해당 시간에 다른 일정이 있습니다. 그래도 예약을 진행할까요?"
                
                blocks = [
                    {"type": "section", "text": {"type": "mrkdwn", "text": confirmation_text}},
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "예 (진행)"},
                                "style": "primary",
                                "value": json.dumps(action_value),
                                "action_id": "confirm_book_anyway"
                            },
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "아니오 (취소)"},
                                "style": "danger",
                                "action_id": "deny_book"
                            }
                        ]
                    }
                ]
                await respond(blocks=blocks, text=confirmation_text, replace_original=True)
                return

            requester_email = action_value["requester_email"]
            result = await asyncio.to_thread(self._create_calendar_event, parsed_data, requester_email, attendee_emails_map)
            # --- 수정된 부분 ---
            # 채널에 전체 메시지를 보내는 대신, 기존 임시 메시지를 최종 결과로 업데이트합니다.
            await respond(text=result, replace_original=True)
            # await client.chat_postMessage(channel=body["channel"]["id"], text=result) # 이 줄을 삭제/주석 처리
        except Exception as e:
            logging.error(f"Error confirming booking: {e}", exc_info=True)
            await respond(text="❌ 예약 확인 중 오류가 발생했습니다.", replace_original=True)
    
    async def _handle_confirm_book_anyway(self, ack, body, client, respond):
        await ack()
        try:
            action_value = json.loads(body["actions"][0]["value"])
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"JSON parsing error in confirm_book_anyway: {e}")
            await respond(text="❌ 요청 처리 중 오류가 발생했습니다.", replace_original=True)
            return

        await respond(text="요청대로 예약을 진행합니다...", replace_original=True)
        
        try:
            parsed_data = action_value["parsed_data"]
            requester_email = action_value["requester_email"]
            attendee_emails_map = action_value["attendee_emails_map"]
            
            result = await asyncio.to_thread(self._create_calendar_event, parsed_data, requester_email, attendee_emails_map)

            # --- 수정된 부분 ---
            # 채널에 전체 메시지를 보내는 대신, 기존 임시 메시지를 최종 결과로 업데이트합니다.
            await respond(text=result, replace_original=True)
            # await client.chat_postMessage(channel=body["channel"]["id"], text=result) # 이 줄을 삭제/주석 처리
        
        except Exception as e:
            logging.error(f"Error confirming booking anyway: {e}", exc_info=True)
            await respond(text="❌ 예약 처리 중 오류가 발생했습니다.", replace_original=True)

    async def _handle_deny_book(self, ack, body, respond):
        await ack()
        await respond(text="예약 요청이 취소되었습니다.", replace_original=True)

    async def _find_available_rooms(self, start_time, end_time):
        try:
            service = self._get_calendar_service()
            
            items_to_check = [{"id": cal_id} for cal_id in self.CALENDAR_RESOURCE_IDS.values()]
            body = {"timeMin": start_time, "timeMax": end_time, "items": items_to_check}
            
            freebusy_result = service.freebusy().query(body=body).execute()
            
            available_rooms = []
            for room_name, cal_id in self.CALENDAR_RESOURCE_IDS.items():
                if not freebusy_result['calendars'][cal_id]['busy']:
                    available_rooms.append(room_name)
            
            return available_rooms, None
        except HttpError as error:
            logging.error(f'Google Calendar API error: {error}')
            return None, f"❌ Google Calendar API 오류: {error}"
        except Exception as e:
            logging.error(f"회의실 검색 중 오류: {e}", exc_info=True)
            return None, "❌ 회의실 검색 중 알 수 없는 오류 발생"

    async def _process_find_room(self, command, respond):
        user_text = command.get("text", "").strip()
        if not user_text:
            await respond(text="❌ 시간을 입력해주세요. 예: `/빈회의실 오늘 오후 3시부터 4시까지`")
            return
        
        try:
            await respond(text=f"🤔 '{user_text}'에 예약 가능한 회의실을 찾고 있습니다...")
            
            parsed_data = await self._parse_booking_request(user_text)
            if not parsed_data or not parsed_data.get('start_time') or not parsed_data.get('end_time'):
                await respond("❌ 요청을 이해하지 못했습니다. 날짜와 시간을 명확하게 다시 요청해주세요.")
                return

            start_time = parsed_data['start_time']
            end_time = parsed_data['end_time']
            
            available_rooms, error = await self._find_available_rooms(start_time, end_time)

            if error:
                await respond(text=error)
                return
            
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            start_str = start_dt.strftime('%#m월 %#d일 %p %#I:%M').replace('AM', '오전').replace('PM', '오후')
            end_str = end_dt.strftime('%p %#I:%M').replace('AM', '오전').replace('PM', '오후')

            if available_rooms:
                result_text = f"✅ *{start_str} ~ {end_str}* 에 예약 가능한 회의실입니다.\n- " + "\n- ".join(available_rooms)
            else:
                result_text = f"❌ 아쉽지만 해당 시간에는 예약 가능한 회의실이 없습니다."

            await respond(text=result_text)

        except Exception as e:
            logging.error(f"Error in /빈회의실 command: {e}", exc_info=True)
            await respond(text="❌ 빈 회의실을 찾는 중 오류가 발생했습니다.")
    
    async def _handle_find_room_command(self, ack, respond, command):
        await ack()
        asyncio.create_task(self._process_find_room(command, respond))

    async def _cleanup_declined_events(self):
        logging.info("🧹 Starting scheduled job: Cleanup declined events...")
        try:
            service = self._get_calendar_service()
            bot_email = service.calendarList().get(calendarId='primary').execute().get('id')
            
            now = datetime.utcnow()
            time_min = now.isoformat() + 'Z'
            time_max = (now + timedelta(days=14)).isoformat() + 'Z'
            
            events_result = service.events().list(
                calendarId='primary', 
                timeMin=time_min, 
                timeMax=time_max,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            for event in events:
                organizer = event.get('organizer', {})
                if organizer.get('email') != bot_email:
                    continue

                attendees = event.get('attendees', [])
                human_attendees = [p for p in attendees if not p.get('resource') and p.get('email') != bot_email]
                
                if human_attendees and all(p.get('responseStatus') == 'declined' for p in human_attendees):
                    event_summary = event.get('summary', '(제목 없음)')
                    event_id = event['id']
                    
                    try:
                        service.events().delete(calendarId='primary', eventId=event_id, sendUpdates='none').execute()
                        logging.info(f"🗑️ Deleted event '{event_summary}' ({event_id}) because all attendees declined.")
                    except HttpError as e:
                        logging.error(f"Failed to delete event {event_id}: {e}")

        except Exception as e:
            logging.error(f"Error in cleanup_declined_events job: {e}", exc_info=True)
    
    async def _check_attendee_availability(self, parsed_data: dict, attendee_emails_map: Dict[str, str]) -> Dict:
        try:
            service = self._get_calendar_service()
            start_time = parsed_data.get("start_time")
            end_time = parsed_data.get("end_time")

            if not attendee_emails_map:
                return {"busy_attendees": [], "error": "조회할 사용자가 없습니다."}

            calendars_to_check = [{"id": email} for email in attendee_emails_map.keys()]
            body = {"timeMin": start_time, "timeMax": end_time, "items": calendars_to_check}

            freebusy_result = service.freebusy().query(body=body).execute()

            busy_attendees = []
            for email, name in attendee_emails_map.items():
                if freebusy_result['calendars'].get(email, {}).get('busy'):
                    busy_attendees.append(name)

            return {"busy_attendees": busy_attendees, "error": None}
        except HttpError as e:
            logging.error(f"Google Calendar API error during schedule check: {e}")
            return {"busy_attendees": [], "error": f"❌ Google Calendar API 오류가 발생했습니다: {e}"}
        except Exception as e:
            logging.error(f"Error checking attendee availability: {e}", exc_info=True)
            return {"busy_attendees": [], "error": "❌ 일정 조회 중 알 수 없는 오류가 발생했습니다."}
            
    async def _process_schedule_check(self, command: Dict, respond):
        user_text = command.get("text", "").strip()
        if not user_text:
            await respond(text="조회할 사용자와 시간을 입력해주세요. 예: `/일정 @사용자 오늘 3시`")
            return

        mentioned_user_ids = re.findall(r'<@(\w+)>', user_text)
        non_standard_mentions = [match.strip() for match in re.findall(r'@([\w.-]+)', user_text)]
        
        time_text = re.sub(r'<@\w+\s*>', '', user_text)
        time_text = re.sub(r'@[\w.-]+\s*', '', time_text).strip()

        if not mentioned_user_ids and not non_standard_mentions:
            await respond(text="일정을 조회할 사용자를 멘션해주세요.")
            return

        if not time_text:
            await respond(text="조회할 시간을 입력해주세요.")
            return

        await respond(text=f"🤔 '{user_text}' 일정을 조회 중입니다...")

        try:
            attendee_emails_map = {}
            all_attendee_names = []
            
            if mentioned_user_ids:
                for mentioned_id in mentioned_user_ids:
                    try:
                        info = await self.app.client.users_profile_get(user=mentioned_id)
                        email = info.get("profile", {}).get("email")
                        name = info.get("profile", {}).get("real_name_normalized", f"@{mentioned_id}")
                        if email and email not in attendee_emails_map:
                            attendee_emails_map[email] = name
                    except SlackApiError as e:
                        logging.warning(f"Failed to get info for user {mentioned_id}: {e.response['error']}")
            
            if non_standard_mentions:
                for name in non_standard_mentions:
                    _, email, real_name = await self._lookup_user_by_name(name)
                    if email and email not in attendee_emails_map:
                        attendee_emails_map[email] = real_name

            if not attendee_emails_map:
                await respond(text="멘션된 사용자의 이메일 정보를 찾을 수 없습니다.")
                return

            parsed_data = await self._parse_booking_request(time_text)
            if not parsed_data or not parsed_data.get('start_time'):
                await respond(text="시간을 정확히 인식하지 못했습니다. 날짜와 시간을 명확하게 다시 입력해주세요. (예: 오늘 3시, 내일 14:00-15:00)")
                return
            
            availability = await self._check_attendee_availability(parsed_data, attendee_emails_map)

            if availability['error']:
                await respond(text=availability['error'])
            elif not availability['busy_attendees']:
                await respond(text="✅ 해당 시간에 모든 인원이 참석 가능합니다.")
            else:
                busy_names = ", ".join(availability['busy_attendees'])
                await respond(text=f"⚠️ {busy_names}님은 해당 시간에 다른 일정이 있습니다.")

        except Exception as e:
            logging.error(f"Error in /일정 command: {e}", exc_info=True)
            await respond(text="❌ 일정 조회 중 오류가 발생했습니다.")
            
    async def _handle_schedule_check_command(self, ack, respond, command):
        await ack()
        asyncio.create_task(self._process_schedule_check(command, respond))

    async def _restart_socket(self):
        logging.info("🔁 Restarting SocketMode connection...")
        if self.socket_handler:
            await self.socket_handler.close()
        self.socket_handler = AsyncSocketModeHandler(self.app, self.SLACK_APP_TOKEN)
        await self.socket_handler.start_async()

    async def start(self) -> None:
        if not all([self.vectorstore, self.retriever, self.llm]):
            raise RuntimeError("Components not properly initialized.")
        logging.info("🚀 Slack bot starting (Socket Mode)...")
        self.register_handlers()
        self.scheduler.start()
        self.socket_handler = AsyncSocketModeHandler(self.app, self.SLACK_APP_TOKEN)
        await self.socket_handler.start_async()

if __name__ == "__main__":
    try:
        bot = SlackBot()
        asyncio.run(bot.start())
    except Exception:
        logging.critical("❌ Failed to start bot", exc_info=True)