'''
    pip install -U langchain langchain-core langchain-commnity langchain-experimental langchain-huggingface langchain-ollama
    pip install streamlit --upgrade
    pip install sentence-transformers
    pip install pdfplumber faiss-cpu pydantic
'''
# streamlit run 7ollama-rag.py

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

# 색상 팔레트 정의
primary_color = "#1E90FF"  # 기본 색상
secondary_color = "#FF6347"  # 보조 색상
background_color = "#F5F5F5"  # 배경 색상
text_color = "#4561e9"  # 텍스트 색상

# 사용자 정의 CSS 적용
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    </style>
""", unsafe_allow_html=True)

# Streamlit 앱 제목 설정
st.title("Ollama 기반 RAG 시스템 구축")

# PDF 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file is not None:
    # 업로드된 파일을 임시 위치에 저장
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # PDF 로더 초기화
    #loader = PDFPlumberLoader("temp.pdf")
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # 문서 분할기 초기화
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # 임베딩 모델 초기화
    embedder = HuggingFaceEmbeddings()

    # 벡터 스토어 생성 및 임베딩 추가
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # LLM 정의
    #llm = ChatOllama(model="deepseek-r1:1.5b")
    llm = ChatOllama(model="qwen3:1.7b")

    # # 시스템 프롬프트 정의
    # system_prompt = (
    # "주어진 문맥을 참고하여 질문에 답하세요. "
    # "답을 모를 경우, '모르겠습니다'라고만 답하고 스스로 답을 만들지 마세요. "
    # "답변은 최대 3문장으로 간결하게 작성하세요. "
    # "최종 답변은 무조건 한국어(korean)으로 작성해주세요"
    # "문맥: {context}"
    # )

    system_prompt = (
        "Answer the question with reference to the given context."
        "If you don't know the answer, just say 'I don't know' and don't make up your own answer."
        "Please write your answer concisely, with a maximum of 3 sentences."
        "Please write your final answer in Korean (Korean) without fail."
        "Context: {context}"
    )

    # ChatPromptTemplate 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 문서 결합 체인 생성 ( Model 과 Prompt 연결)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # 검색 기반 QA 체인 생성 ( prompt + model + retriever )
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 사용자 입력 받기
    user_input = st.text_input("PDF와 관련된 질문을 입력하세요:")

    # 사용자 입력 처리
    if user_input:
        with st.spinner("처리 중..."):
            response = rag_chain.invoke({"input": user_input})
            st.write("응답:")
            st.write(response.get("answer","응답을 처리할 수 없습니다."))
else:
    st.write("진행하려면 PDF 파일을 업로드하세요.")