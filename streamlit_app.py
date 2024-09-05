import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
import tempfile
import os


st.title("📄 Assignment 15")
st.write(
    "문서를 업로드하면 GPT가 문서를 읽고 답을 드려요"
    "OpenAPI 사용을 위해 Settings에 API Key를 입력해주세요"
)
cache_dir = LocalFileStore("./.cache/")

with st.sidebar:
    st.title("Settings")
    st.markdown("***")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")

    uploaded_file = st.file_uploader(
        "문서 파일을 업로드해주세요 (.txt, .pdf, .docx)", type=("txt", "pdf", "docx")
    )
    st.markdown("***")
    st.link_button(
        "Github Repo 바로가기", "https://github.com/asuracoder91/streamlit_rag"
    )

question = st.text_area(
    "문서에 대해 질문을 하세요",
    placeholder="문서 요약좀 해줄래?",
    disabled=not uploaded_file,
)

if openai_api_key and uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.1)

    # 파일 형식에 맞는 로더 선택
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension == ".pdf":
        loader = PyPDFLoader(tmp_file_path)  # PDF 파일 처리
    elif file_extension == ".txt":
        loader = TextLoader(tmp_file_path)  # 텍스트 파일 처리
    elif file_extension == ".docx":
        loader = UnstructuredFileLoader(tmp_file_path)  # DOCX 파일 처리
    else:
        st.error("지원되지 않는 파일 형식입니다.")
        st.stop()

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{context}, and say in Korean Only.",
            ),
            ("human", "{question}"),
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def chain_with_memory(question):
        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        inputs = {
            "context": context,
            "question": question,
            "chat_history": chat_history,
        }

        prompt = prompt_template.format_messages(**inputs)
        response = llm(prompt)

        memory.save_context({"question": question}, {"answer": response.content})

        return response.content

    if st.button("질문에 답변 받기"):
        with st.spinner("GPT가 문서를 읽고 답변 중입니다..."):
            answer = chain_with_memory(question)
            st.write("### 답변:")
            st.write(answer)

# 오류 또는 알림 처리
elif not openai_api_key:
    st.warning("OpenAI API 키를 입력해주세요.")
elif not uploaded_file:
    st.info("문서를 업로드해주세요.")
elif not question:
    st.info("문서에 대해 질문을 입력해주세요.")
