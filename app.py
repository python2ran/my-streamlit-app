
import os, json, hashlib
import streamlit as st
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText

# 히스토리 개수 제한
MAX_HISTORY = 10

# 이메일 수신자 (김 대리 본인)
MY_EMAIL = "python2ran@gmail.com"    # 수정 필요

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="AI 개인비서", page_icon="🤖", layout="wide")
st.title("🤖 AI 개인비서 AIVLE")
st.write("안녕하세요? AI 개인비서 에이블입니다. 음성이나 텍스트로 업무를 지시하세요.")


# ------------------------------------------------------------
# 도구 정의
# ------------------------------------------------------------
tools = [{
    "type": "function",
    "name": "send_email",
    "description": "요약 내용이나 업무 정리 내용을 이메일로 발송합니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "이메일 제목. 예: '마케팅 전략 회의 요약'"
            },
            "body": {
                "type": "string",
                "description": "이메일 본문. 예: '안녕하세요? 김대리입니다.'"
            }
        },
        "required": ["subject", "body"],
        "additionalProperties": False
    },
    "strict": True
}]


# ------------------------------------------------------------
# 기능 함수
# ------------------------------------------------------------
# 이메일 발송
def send_email(to, subject, body):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")

    # 메일 구성 (본문, 제목, 발신자, 수신자)
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    # 메일 보내기
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success", "subject": subject}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# STT: 음성 → 텍스트
def speech_to_text(audio_bytes):
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_bytes, "audio/wav")
    )
    return transcript.text


# TTS: 텍스트 → 음성
def text_to_speech(text):
    speech = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text[:4000]    # TTS 입력 길이 제한(4096자)
    )
    return speech.content


# 업무 목록 추출 (JSON 형식으로 응답 요청)
def extract_tasks(document_text):
    prompt = f"""
    다음 회의록에서 김 대리에게 배정된 업무만 뽑아내세요.
    설명 없이 JSON 배열로만 답하세요.
    형식: ["업무1", "업무2"]
    배정된 업무가 없으면 [] 로 답하세요.

    [회의록]
    {document_text}
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}]
    )
    answer = response.output_text.strip().replace("```json", "").replace("```", "")
    return json.loads(answer)


# 역할 지정
sys_role = f"""
당신은 '김 대리'의 개인비서 '에이블'입니다.
김 대리가 요청한 업무를 최선을 다해 처리합니다.
답변은 실제 비서가 답변하듯 정중해야 합니다.

[담당 업무]
- 미팅 내용 요약
- 김 대리 업무 파악
- 이메일 작성 및 발송
- 번역
- 기타 질문에 대한 답변

[답변 형식]
- 미팅 내용 요약을 요청받으면 다음 순서로 정리해 답변합니다.
  1) 핵심 논의 내용
  2) 결정 사항
  3) 김 대리 담당 업무
- 그 외 질문은 간결하게 답변합니다.

[이메일 정보]
- 이메일은 김 대리 본인({MY_EMAIL})에게 발송합니다.
- 이메일로 보내달라는 지시가 있을 때만 발송합니다.
"""


# ------------------------------------------------------------
# 상태 초기화
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "tasks" not in st.session_state:
    st.session_state.tasks = []

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = ""


# ------------------------------------------------------------
# 사이드바
# ------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/Jangrae/img/master/ai_robot.png")
    st.markdown(
        "<h3 style='text-align: center;'>에이블이 도와드립니다</h3>",
        unsafe_allow_html=True
    )

    # 문서 업로드
    st.divider()
    st.subheader("📄 회의록 업로드")
    uploaded_file = st.file_uploader("텍스트 파일", type=["txt"])

    if uploaded_file is not None:
        text = uploaded_file.getvalue().decode("utf-8")   # read()는 두 번째 실행 시 빈 값 반환
        if st.session_state.document_name != uploaded_file.name:
            st.session_state.document_text = text
            st.session_state.document_name = uploaded_file.name
            st.session_state.tasks = []

    # 업로드된 문서 정보 표시
    if st.session_state.document_text:
        st.success(f"✅ {st.session_state.document_name}")
        st.caption(f"글자 수: {len(st.session_state.document_text):,}자")
        with st.expander("문서 내용 미리보기"):
            st.text(st.session_state.document_text[:300] + " ...")

        # 업무 목록 추출
        if st.button("📋 내 업무 목록 뽑기"):
            with st.spinner("업무를 정리하고 있습니다..."):
                try:
                    st.session_state.tasks = extract_tasks(st.session_state.document_text)
                except Exception as e:
                    st.error(f"업무 목록을 만들지 못했습니다: {e}")

    # 업무 체크리스트
    if st.session_state.tasks:
        st.divider()
        st.subheader("✅ 김 대리 업무")
        for i, task in enumerate(st.session_state.tasks):
            st.checkbox(task, key=f"task_{i}")

    # 사용법
    st.divider()
    with st.expander("💡 사용법"):
        st.markdown("""
        1. 회의록(txt)을 업로드합니다.
        2. 마이크로 녹음하거나 아래에 직접 입력합니다.
        3. 예시 지시
            - "회의 내용을 요약해줘"
            - "내가 할 일을 알려줘"
            - "요약한 내용을 이메일로 보내줘"
        """)

    # 대화 초기화
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.session_state.last_audio_hash = ""
        st.rerun()

    st.caption("본 서비스는 김 대리 전용 개인비서입니다.")


# ------------------------------------------------------------
# 대화 내용 출력
# ------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ------------------------------------------------------------
# 사용자 입력 (음성 또는 텍스트)
# ------------------------------------------------------------
user_input = None

# 음성 입력
audio_value = st.audio_input("🎙️ 음성으로 업무를 지시하세요.")

if audio_value is not None:
    audio_bytes = audio_value.getvalue()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()

    # 같은 녹음이 다시 처리되지 않도록 확인
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with st.status("음성을 텍스트로 변환하고 있습니다...", expanded=False):
            try:
                user_input = speech_to_text(audio_bytes)
            except Exception as e:
                st.error(f"음성 인식에 실패했습니다: {e}")

# 텍스트 입력
text_input = st.chat_input("메시지를 입력하세요.")
if text_input:
    user_input = text_input


# ------------------------------------------------------------
# 업무 처리
# ------------------------------------------------------------
if user_input:
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 시스템 메시지 구성 (업로드한 문서 포함)
    system_content = sys_role
    if st.session_state.document_text:
        system_content += f"\n\n[참고 문서]\n{st.session_state.document_text}"

    # 시스템 메시지 + 최근 대화 히스토리
    input_messages = [{"role": "system", "content": system_content}]
    input_messages += st.session_state.messages[-MAX_HISTORY:]

    with st.chat_message("assistant"):
        answer = ""
        email_result = None

        with st.status("답변을 준비하고 있습니다...", expanded=False) as status:
            try:
                # LLM 호출
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=input_messages,
                    tools=tools,
                    tool_choice="auto"
                )

                # 도구 필요 여부 확인
                tool_calls = [item for item in response.output
                              if item.type == "function_call"]

                if not tool_calls:
                    answer = response.output_text
                else:
                    # LLM이 요청한 도구 실행
                    status.update(label="이메일을 발송하고 있습니다...")
                    tool_outputs = []
                    for item in tool_calls:
                        args = json.loads(item.arguments)
                        if item.name == "send_email":
                            email_result = send_email(to=MY_EMAIL, **args)
                            tool_outputs.append({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": json.dumps(email_result, ensure_ascii=False)
                            })

                    # 도구 수행 결과를 LLM에 다시 전달
                    status.update(label="답변을 정리하고 있습니다...")
                    final_response = client.responses.create(
                        model="gpt-4o-mini",
                        previous_response_id=response.id,
                        input=tool_outputs
                    )
                    answer = final_response.output_text

                status.update(label="완료", state="complete")

            except Exception as e:
                status.update(label="처리 실패", state="error")
                st.error(f"요청을 처리하지 못했습니다: {e}")

        # 답변 출력
        if answer:
            st.markdown(answer)

            # 이메일 발송 결과 표시
            if email_result is not None:
                if email_result["status"] == "success":
                    st.success(f"📧 이메일을 발송했습니다. (제목: {email_result['subject']})")
                else:
                    st.error(f"📧 이메일 발송에 실패했습니다: {email_result['error']}")

            # TTS: 텍스트 → 음성
            try:
                audio_answer = text_to_speech(answer)
                st.audio(audio_answer, format="audio/mp3", autoplay=True)
            except Exception as e:
                st.warning(f"음성 생성에 실패했습니다: {e}")

            # 대화 기록 저장
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
