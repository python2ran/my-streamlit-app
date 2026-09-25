
import os
import tempfile
import hashlib
import streamlit as st

from operator import itemgetter

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 유틸
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 파이프라인 함수 (01번 노트북과 동일)
def rag_pipeline(filepath):

    # PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""])
    chunks = splitter.split_documents(docs)

    # Embedding 모델
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    # 벡터저장소 만들기
    vectorstore = Chroma(
        collection_name="aivle_docs",
        embedding_function=embedding,
        persist_directory="chroma_db"
    )

    # 기존 데이터 비우기
    vectorstore.reset_collection()

    # 문서 추가
    _ = vectorstore.add_documents(chunks)

    # 검색기 구성
    retriever = vectorstore.as_retriever()

    # LLM 선언
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 프롬프트 구성
    s_msg = """
            당신은 AIVLE School 문의에 답하는 학습 도우미입니다.
            질문의 대부분은 AIVLE School에 참여를 고민하는 학생들의 질문입니다.

            규칙:
            1. 반드시 제공된 문서 내용(context)에 근거하여 답변하세요.
            2. 문서에 없는 내용은 추측하지 말고 "업로드된 문서에서 확인되지 않습니다."라고 답하세요.
            3. 답변은 학습 도전을 격려하는 형식으로 친절하게 작성하세요.

            [문서]
            {context}
            """
    h_msg = "{input}"
    messages = [("system", s_msg), ("human", h_msg)]
    prompt = ChatPromptTemplate.from_messages(messages)

    # RAG 체인 구성
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Streamlit UI
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="📘")
st.markdown("""
## 📘 AIVLE School 학습 도우미
여러분의 도전을 진심으로 응원합니다 🚀
""")

# 상태 초기화
if "chain" not in st.session_state:
    st.session_state.chain = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바
with st.sidebar:
    st.image("https://raw.githubusercontent.com/Jangrae/img/master/aivle.png")
    st.text("에이블스쿨은 기업 실무형 AI/DX인재를 양성하는 교육 프로그램 입니다.")

    # 관리자 영역
    st.divider()
    st.header("관리자")

    uploaded_file = st.file_uploader("PDF 업로드", type=["pdf"])
    if uploaded_file:
        if st.session_state.document_name != uploaded_file.name:
            with st.spinner("문서를 처리하고 있습니다..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                st.session_state.chain = rag_pipeline(tmp_path)
                st.session_state.document_name = uploaded_file.name
                st.session_state.messages = []

    if st.session_state.document_name:
        st.success("✅ 문서 활성화됨")
    else:
        st.warning("⚠️ 문서 없음")

    # 추천 질문
    st.divider()
    st.subheader("💡 추천 질문")

    if st.button("미니 프로젝트 설명"):
        st.session_state.pending_query = "미니 프로젝트에 대해 알려주세요."
        st.rerun()

    if st.button("무엇을 배우나요?"):
        st.session_state.pending_query = "AIVLE에서 배우는 내용은 무엇인가요?"
        st.rerun()

    # 대화 초기화 버튼
    st.divider()
    if st.button("💬 대화 초기화"):
        st.session_state.messages = []
        st.session_state.init_message = True
        st.rerun()

# 기존 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 질문 입력
if st.session_state.chain is None:
    st.chat_input("아직 답변할 준비가 되지 않았습니다.", disabled=True)
    query = None
else:
    query = st.chat_input("질문을 입력하세요")

# 버튼에서 들어온 query 처리
if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            if st.session_state.chain is None:
                raw_answer = "아직 답변할 준비가 되지 않았습니다."
            else:
                raw_answer = st.session_state.chain.invoke({"input": query})

            # 스트리밍 효과
            placeholder = st.empty()
            answer = ""
            for char in raw_answer:
                answer += char
                placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
