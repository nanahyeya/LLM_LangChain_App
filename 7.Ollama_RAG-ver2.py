# streamlit run 7ollama-rag-ver2.py

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# =====================
#  UI 스타일 설정
# =====================
PRIMARY_COLOR = "#1E90FF"
SECONDARY_COLOR = "#FF6347"
BACKGROUND_COLOR = "#F5F5F5"
TEXT_COLOR = "#333333"
SUCCESS_COLOR = "#DFF2BF"
INFO_COLOR = "#BDE5F8"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {PRIMARY_COLOR};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {SECONDARY_COLOR};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stAlert-success {{
        background-color: {SUCCESS_COLOR} !important;
        color: {TEXT_COLOR} !important;
        font-weight: bold !important;
        padding: 10px;
        border-radius: 5px;
    }}
    .stAlert-info {{
        background-color: {INFO_COLOR} !important;
        color: {TEXT_COLOR} !important;
        font-weight: bold !important;
        padding: 10px;
        border-radius: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================
#  RAG 시스템 설정
# =====================
#MODEL_NAME = "deepseek-r1:1.5b"
MODEL_NAME = "qwen3:1.7b"
SEARCH_K = 3
TEMP_PDF_PATH = "uploaded_document.pdf"

SYSTEM_PROMPT = (
    "Answer the question with reference to the given context. "
    "If you don't know the answer, just say 'I don't know' and don't make up your own answer. "
    "Please write your answer concisely, with a maximum of 3 sentences. "
    "Please write your final answer in Korean (Korean) without fail. "
    "Context: {context}"
)

# =====================
#  Streamlit UI
# =====================
st.title("Ollama 기반 RAG 시스템")

# 파일 업로드
uploaded_file = st.file_uploader("📄 PDF 파일을 업로드하세요", type="pdf")

if uploaded_file and "rag_chain" not in st.session_state:
    try:
        with st.spinner(" PDF 파일을 처리 중..."):
            # PDF 로드
            with open(TEMP_PDF_PATH, "wb") as f:
                f.write(uploaded_file.getvalue())
                
            loader = PyPDFLoader(TEMP_PDF_PATH)
            docs = loader.load()

            # 문서 분할
            embedder = HuggingFaceEmbeddings()
            text_splitter = SemanticChunker(embedder)
            documents = text_splitter.split_documents(docs)

            # 벡터 저장소 생성
            vector_store = FAISS.from_documents(documents, embedder)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": SEARCH_K})

            # LLM 및 체인 구성
            llm = ChatOllama(model=MODEL_NAME)
            prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

            st.markdown('<p class="stAlert-success"> PDF 처리가 완료되었습니다! 이제 질문을 입력하세요.</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")

# 사용자 입력 처리
user_input = st.text_input(" PDF와 관련된 질문을 입력하세요:")

if user_input and "rag_chain" in st.session_state:
    with st.spinner(" 답변 생성 중..."):
        try:
            response = st.session_state.rag_chain.invoke({"input": user_input})
            st.subheader(" 응답:")
            st.write(response.get("answer", " 응답을 생성할 수 없습니다."))
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {e}")
elif not uploaded_file:
    st.markdown('<p class="stAlert-info"> 진행하려면 PDF 파일을 업로드하세요.</p>', unsafe_allow_html=True)