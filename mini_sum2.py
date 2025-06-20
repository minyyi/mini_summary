# ============== PDF 분석 Streamlit 앱 ==============
import streamlit as st
import fitz  # pymupdf
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List, Dict, Any
import pickle
import os
import tempfile
from dotenv import load_dotenv
import time

# 페이지 설정
st.set_page_config(
    page_title="PDF 분석 AI 어시스턴트",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 환경 설정
load_dotenv()

# ============== 세션 상태 초기화 ==============
def init_session_state():
    """세션 상태 안전하게 초기화"""
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'metadatas' not in st.session_state:
        st.session_state.metadatas = []
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_input_text' not in st.session_state: # text_input의 key와 동일하게 설정
        st.session_state.user_input_text = ""
    if 'last_processed_file_name' not in st.session_state:
        st.session_state.last_processed_file_name = None
    # Langgraph 그래프 인스턴스를 세션 상태에 저장
    if 'agent_graph' not in st.session_state:
        st.session_state.agent_graph = None

# 세션 상태 초기화 함수 호출
init_session_state()

# ============== Langchain 설정 ==============
# OpenAI API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY 환경 변수를 설정해주세요.")
    st.stop()

# LLM 및 임베딩 모델 초기화 (세션 상태에 저장)
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.3)

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-large")

st.session_state.llm = get_llm()
st.session_state.embedding_model = get_embedding_model()

# ============== FAISS 인덱싱 및 검색 함수 ==============
def create_faiss_index(chunks, metadatas):
    """주어진 텍스트 덩어리들로 FAISS 인덱스를 생성하고 반환합니다."""
    if not chunks:
        return None
    try:
        embeddings = st.session_state.embedding_model.embed_documents(chunks)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings).astype('float32'))
        return index
    except Exception as e:
        st.error(f"FAISS 인덱스 생성 중 오류: {e}")
        return None

def search_faiss_index(query, index, chunks, metadatas, k=3):
    """FAISS 인덱스에서 쿼리와 가장 유사한 k개의 덩어리를 검색합니다."""
    if index is None:
        return []
    try:
        query_embedding = st.session_state.embedding_model.embed_query(query)
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        results = []
        for i in I[0]:
            if i != -1:  # 유효한 인덱스만 추가
                results.append({"content": chunks[i], "metadata": metadatas[i]})
        return results
    except Exception as e:
        st.error(f"FAISS 인덱스 검색 중 오류: {e}")
        return []

# ============== 텍스트 분할 함수 ==============
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """텍스트를 덩어리로 분할합니다."""
    # 간단한 텍스트 분할 로직 (개선을 위해 Langchain의 RecursiveCharacterTextSplitter 사용 고려)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

# ============== PDF 처리 함수 ==============
def process_pdf(uploaded_file):
    """PDF 파일을 처리하여 텍스트 덩어리와 메타데이터를 추출합니다."""
    if uploaded_file is None:
        return [], []

    # 파일 이름이 바뀌지 않았고 이미 처리된 경우 다시 처리하지 않음
    if (st.session_state.file_processed and
        st.session_state.last_processed_file_name == uploaded_file.name):
        return st.session_state.chunks, st.session_state.metadatas

    try:
        # Streamlit의 UploadedFile 객체에서 직접 파일 내용 읽기
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        full_text = ""
        page_texts = []
        page_metadatas = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            full_text += text
            page_texts.append(text) # 페이지별 텍스트 저장

            # 간단한 메타데이터
            page_metadatas.append({"page_number": page_num + 1, "source": uploaded_file.name})

        # 전체 텍스트를 덩어리로 분할 (RAG 시스템을 위해)
        chunks = get_text_chunks(full_text)
        metadatas_for_chunks = []
        # 각 덩어리에 해당하는 페이지 정보 추정 (간단화)
        # 실제 구현에서는 텍스트 스플리터가 메타데이터를 포함하도록 해야 함
        for i, chunk in enumerate(chunks):
            metadatas_for_chunks.append({"chunk_index": i, "source": uploaded_file.name})

        st.session_state.chunks = chunks
        st.session_state.metadatas = metadatas_for_chunks
        st.session_state.file_processed = True
        st.session_state.last_processed_file_name = uploaded_file.name
        return chunks, metadatas_for_chunks
    except Exception as e:
        st.error(f"PDF 처리 중 오류: {e}")
        st.session_state.file_processed = False
        st.session_state.last_processed_file_name = None
        return [], []

# ============== Langgraph 툴 정의 및 그래프 생성 ==============
@tool
def pdf_search_tool(query: str):
    """
    업로드된 PDF 문서에서 사용자의 질문과 관련된 정보를 검색합니다.
    사용자의 질문이 특정 문서 내용을 요구할 때 이 도구를 사용하세요.
    예시: "이 문서에서 주요 내용은 뭐야?", "PDF에 따르면 A에 대한 정보는 뭐야?"
    """
    if st.session_state.index is None:
        return "PDF 파일이 업로드되지 않았거나 인덱스가 생성되지 않았습니다."
    
    results = search_faiss_index(query, st.session_state.index, st.session_state.chunks, st.session_state.metadatas)
    
    if not results:
        return "죄송합니다. PDF에서 관련 정보를 찾을 수 없습니다."
    
    # 검색된 내용을 요약하여 반환
    context = "\n\n".join([r["content"] for r in results])
    
    # LLM을 사용하여 검색된 내용을 바탕으로 답변 생성
    prompt = f"""
    다음은 PDF 문서에서 검색된 관련 정보입니다:
    ---
    {context}
    ---
    이 정보를 바탕으로 사용자의 질문에 답해주세요.
    질문: {query}
    """
    
    try:
        response = st.session_state.llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

@tool
def general_chat_tool(query: str):
    """
    업로드된 PDF와 관련 없는 일반적인 질문에 답하거나, PDF 검색 결과로 충분한 답변을 생성할 수 없을 때 사용합니다.
    이 도구는 일반 상식, 정의, 개념 설명 등 넓은 범위의 질문에 답할 수 있습니다.
    예시: "오늘 날씨 어때?", "파이썬이란?", "인공지능의 종류는?"
    """
    try:
        response = st.session_state.llm.invoke(query)
        return response.content
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

# Langgraph 노드 정의
def call_model(state):
    messages = state["messages"]
    last_message = messages[-1]

    # 마지막 메시지가 ToolMessage인 경우, 툴 출력으로 응답 생성
    if isinstance(last_message, ToolMessage):
        # 툴 실행 결과를 바탕으로 응답 생성 프롬프트 구성
        tool_output = last_message.content
        human_message_content = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                human_message_content = msg.content
                break # 첫 번째 HumanMessage를 찾으면 종료

        prompt = f"""
        사용자의 원래 질문: {human_message_content}
        도구 검색 결과: {tool_output}
        
        이 정보를 바탕으로 사용자에게 친절하고 명확하게 답변해주세요.
        """
        response = st.session_state.llm.invoke(prompt)
        return {"messages": messages + [AIMessage(content=response.content)]}
    else:
        # AI 모델이 툴 사용 여부를 결정하도록 함
        response = st.session_state.llm.invoke(messages)
        return {"messages": messages + [response]}

# Langgraph 상태 정의
class AgentState(TypedDict):
    messages: List[Any]

# Langgraph 그래프 구축 함수
@st.cache_resource
def create_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("call_model", call_model)
    workflow.add_node("tool_node", ToolNode(tools=[pdf_search_tool, general_chat_tool]))

    workflow.set_entry_point("call_model")

    def route_tools(state):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        return END

    workflow.add_conditional_edges(
        "call_model",
        route_tools,
        {"tool_node": "tool_node", END: END}
    )
    workflow.add_edge("tool_node", "call_model") # 툴 실행 후 다시 모델 호출 (선택적)
    return workflow.compile()

# 앱 초기화 시 그래프 생성
if st.session_state.agent_graph is None:
    st.session_state.agent_graph = create_agent_graph()

# ============== Streamlit UI ==============
st.title("📚 PDF 분석 AI 어시스턴트")
st.write("PDF 파일을 업로드하고 질문하면 AI가 문서 내용을 기반으로 답변해 드립니다.")

with st.sidebar:
    st.header("PDF 파일 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 여기에 드래그 앤 드롭하거나 클릭하여 업로드하세요.", type="pdf")

    if uploaded_file:
        st.write(f"업로드된 파일: {uploaded_file.name}")
        with st.spinner("PDF를 처리 중입니다..."):
            chunks, metadatas = process_pdf(uploaded_file)
            if chunks:
                st.session_state.index = create_faiss_index(chunks, metadatas)
                if st.session_state.index:
                    st.success("PDF 처리 및 인덱싱 완료!")
                else:
                    st.error("FAISS 인덱스 생성에 실패했습니다.")
            else:
                st.error("PDF에서 텍스트를 추출하지 못했습니다.")
    else:
        st.session_state.file_processed = False
        st.session_state.last_processed_file_name = None
        st.session_state.chunks = []
        st.session_state.metadatas = []
        st.session_state.index = None
        # PDF 파일이 없을 경우 기존 채팅 기록을 초기화 (선택 사항)
        # st.session_state.chat_history = []
        st.info("PDF 파일을 업로드하여 시작하세요.")

# 채팅 기록 표시 (st.chat_message 사용)
st.subheader("대화 기록")
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# 사용자 입력 처리 함수
def handle_query_submission():
    current_input = st.session_state.user_input_text.strip()
    if not current_input:
        return # 입력값이 없으면 아무것도 하지 않음

    # 사용자 메시지 채팅 기록에 추가 및 UI 표시
    st.session_state.chat_history.append(("user", current_input))
    
    # AI 응답 생성
    with st.spinner("AI가 답변을 생성하는 중입니다..."):
        try:
            # Langgraph 실행
            initial_state = {"messages": [HumanMessage(content=current_input)]}
            result = st.session_state.agent_graph.invoke(initial_state) # invoke로 변경
            
            # 응답 추출
            if result and "messages" in result:
                final_message = result["messages"][-1]
                response = final_message.content if hasattr(final_message, 'content') else str(final_message)
            else:
                response = "응답을 생성할 수 없습니다."
            
            # AI 메시지 채팅 기록에 추가
            st.session_state.chat_history.append(("assistant", response))
            
        except Exception as e:
            error_msg = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history.append(("assistant", error_msg))
    
    # 입력창 초기화
    st.session_state.user_input_text = ""
    # 이 콜백 함수가 완료되면 Streamlit이 자동으로 전체 스크립트를 재실행하여 UI를 업데이트합니다.
    # 따라서 명시적인 st.rerun()은 필요하지 않으며, 오히려 중복 재실행을 유발할 수 있습니다.


# 사용자 입력 필드 (Enter 키 입력 시 handle_query_submission 호출)
st.text_input(
    "여기에 질문을 입력하고 Enter를 누르세요:",
    key="user_input_text",
    on_change=handle_query_submission, # 엔터 입력 시 콜백
    placeholder="PDF 내용에 대해 질문해주세요..."
)

# "전송" 버튼 (클릭 시 handle_query_submission 호출)
# st.button은 클릭 시 True를 반환하며, 이때 콜백을 실행합니다.
if st.button("전송"):
    # 버튼 클릭 시 handle_query_submission 호출 (st.text_input의 현재 값을 사용)
    handle_query_submission()


# 하단 퀵 질문 버튼 (기존 로직 유지, 직접 툴 호출)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 PDF 검색 질문 (예시)"):
        if st.session_state.index is None:
            st.warning("먼저 PDF 파일을 업로드하고 인덱싱해야 합니다.")
        else:
            query = "보고서의 주요 내용은 무엇인가요?"
            st.session_state.chat_history.append(("user", query))
            with st.spinner("답변을 생성하는 중입니다..."):
                try:
                    response = pdf_search_tool(query) # 직접 툴 호출 (Langgraph 통하지 않음)
                    st.session_state.chat_history.append(("assistant", response))
                except Exception as e:
                    st.error(f"검색 중 오류: {e}")
            st.rerun()
        
with col2:
    if st.button("🌐 일반 질문 (예시)"):
        query = "오늘 날씨는?"
        st.session_state.chat_history.append(("user", query))
        with st.spinner("답변을 생성하는 중입니다..."):
            try:
                response = general_chat_tool(query) # 직접 툴 호출
                st.session_state.chat_history.append(("assistant", response))
            except Exception as e:
                st.error(f"검색 중 오류: {e}")
        st.rerun()
        
with col3:
    if st.button("🤖 AI 질문 (예시)"):
        query = "ChatGPT와 Claude의 차이점은?"
        st.session_state.chat_history.append(("user", query))
        with st.spinner("답변을 생성하는 중입니다..."):
            try:
                response = general_chat_tool(query) # 직접 툴 호출
                st.session_state.chat_history.append(("assistant", response))
            except Exception as e:
                st.error(f"답변 생성 중 오류: {e}")
        st.rerun()
        
with col4:
    if st.button("💻 코딩 질문 (예시)"):
        query = "파이썬 클래스와 객체의 차이점을 설명해줘"
        st.session_state.chat_history.append(("user", query))
        with st.spinner("답변을 생성하는 중입니다..."):
            try:
                response = general_chat_tool(query) # 직접 툴 호출
                st.session_state.chat_history.append(("assistant", response))
            except Exception as e:
                st.error(f"답변 생성 중 오류: {e}")
        st.rerun()