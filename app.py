
import os, json
import streamlit as st
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText

# 히스토리 개수 제한
MAX_HISTORY = 10

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="AI 개인비서", page_icon="🤖", layout="wide")
st.title("🤖 AI 개인비서 AIVLE")
st.write("안녕하세요? AI 개인비서 에이블입니다.")

# 사이드 바에 AI 비서 이미지 표시
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/Jangrae/img/master/ai_robot.png"
    )
    st.markdown(
        "<h3 style='text-align: center;'>음성으로 업무를 지시하세요.</h3>",
        unsafe_allow_html=True
    )

# 도구 정의
tools = [{
    "type": "function",
    "name": "send_email",
    "description": "이메일을 발송합니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "이메일 수신자. 예: 'test@gmail.com'"
            },
            "subject": {
                "type": "string",
                "description": "이메일 제목. 예: '마케팅 전략 초안'"
            },
            "body": {
                "type": "string",
                "description": "이메일 본문. 예: '안녕하세요? 김대리입니다.'"
            }
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False
    },
    "strict": True
}]

# 함수 만들기
def send_email(to, subject, body):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")

    # 메일 구성 (본문, 제목, 발신자, 수신자)
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    # 메일 보내기 (with 구문: 오류가 나도 연결이 닫힘)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

# 역할 지정
sys_role = """
당신은 '김 대리'의 개인 비서 '에이블'입니다.
김대리가 요청한 업무를 최선을 다해 처리합니다.
답변은 실제 비서가 답변하듯 정중해야 합니다.
다음과 같은 업무를 주로 담당하게 됩니다.
[담당 업무]
- 미팅 내용 요약
- 김 대리 업무 파악
- 이메일 작성 및 발송
- 번역
- 기타 질문에 대한 답변
[이메일 정보]
- 이메일은 python2ran@gmail.com으로 발송
"""

# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 업무지시 입력 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("업무지시 실행")

# 버튼을 눌렀을 때만 실행
if submit:
    # 업로드 된 파일이 있으면 디코딩
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    # 녹음이 없으면 녹음 요청
    if audio_value is None:
        st.warning("음성을 녹음해주세요.")
        st.stop()

    # STT: 음성 → 텍스트
    with st.spinner("음성을 텍스트로 변환 중입니다..."):
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", audio_value, "audio/wav")
        )

    # 지시사항 출력 (확인용)
    st.subheader("지시사항")
    st.write(transcript.text)

    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": transcript.text})

    # 최근 10개 히스토리 추출
    history = st.session_state.messages[-MAX_HISTORY:]

    # 시스템 메세지
    system_content = sys_role
    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"
    input_messages = [{"role": "system", "content": system_content}]

    # 시스템 메세지 + 히스토리
    input_messages += history

    with st.spinner("답변을 준비하고 있습니다..."):
        # LLM 호출
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages,
            tools=tools,
            tool_choice="auto"
        )

        # 도구 필요 여부 확인
        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        # 도구가 필요없는 경우
        if not tool_calls:
            # 답변 출력
            answer = response.output_text
        # 도구가 필요한 경우
        else:
            # LLM이 요청한 도구 실행
            tool_outputs = []
            for item in tool_calls:
                if item.type == "function_call":
                    args = json.loads(item.arguments)

                    if item.name == "send_email":
                        try:
                            args["to"] = "python2ran@gmail.com"
                            result = send_email(**args)
                        except Exception as e:
                            result = {"status": "error", "message": str(e)}

                        tool_outputs.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(result, ensure_ascii=False)
                        })

            # 도구 수행 결과를 LLM에 다시 전달
            final_response = client.responses.create(
                model="gpt-4o-mini",
                previous_response_id=response.id,
                input=tool_outputs
            )

            # LLM 최종 답변
            answer = final_response.output_text

        # 응답 저장
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # 최근 히스토리 10개만 저장
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

        # TTS: 텍스트 → 음성
        try:
            speech = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=answer[:4000]    # TTS 입력 길이 제한(4096자)
            )
            audio_bytes = speech.content

            # 음성 출력(자동 재생)
            st.subheader("AI 음성 답변")
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.warning(f"음성 생성에 실패했습니다: {e}")

        # 텍스트 출력 (확인용)
        st.write(answer)
