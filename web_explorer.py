import os
import streamlit as st
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain.chat_models.gigachat import GigaChat
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever


os.environ["GOOGLE_API_KEY"] = "" # ПОлучить тут https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials
os.environ["GOOGLE_CSE_ID"] = "" # Получить тут https://programmablesearchengine.google.com/

GIGACHAT_API_KEY=""

st.set_page_config(page_title="Compas+", page_icon="🧭")

def settings():

    # Vectorstore

    embeddings_model = GigaChatEmbeddings(
        credentials=GIGACHAT_API_KEY,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP")

    embedding_size = 1024
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM
    llm = GigaChat(credentials=GIGACHAT_API_KEY,
                   verify_ssl_certs=False, scope="GIGACHAT_API_CORP")

    # Search
    search = GoogleSearchAPIWrapper()

    # Initialize
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm,
        search=search,
        num_search_results=1,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150),

    )

    return web_retriever, llm

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


st.sidebar.image("img/compas.png")
st.header("`Compas+ KnowledGe NavIGAtor`")

st.info("Compas+, исследователь миров интернета, навигатор знаний, инструмент столь же вдохновляющий, как компас. "
        "Его можно использовать его для навигации по библиотекам, музеям, базам данных или архивам. "
        "Это устройство не просто приведет вас к первоисточникам, оно пригласит вас глубоко внутрь их тайн, интерпретируя и объясняя, "
        "превращая огромные объемы информации в персонализированные и понятные знания. "
        "Система работает на GigaChat API.")

# Make retriever and llm
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever, verbose=True,
                                                           return_source_documents=True)
retrieval_streamer_cb = PrintRetrievalHandler(st.container())


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if question := st.chat_input("Введите свой вопрос"):

    # Generate answer (w/ citations)
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})


    # Write answer and sources

    st_answer = st.empty()
    stream_handler = StreamHandler(st_answer, initial_text="`Ответ:`\n\n")
    try:
        result = qa_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])

        system_answer = result['answer']
        formatted_answer = '`Ответ:`\n\n' + system_answer
        st_answer.info(formatted_answer)

        st.session_state.messages.append({"role": "assistant", "content": formatted_answer})

        if result['source_documents']:
            short_summary = f"{result['answer'].split('.')[0]}."
            sources = result['sources']

            formatted_short_answer = '`Главная идея:`\n\n' + short_summary
            formatted_sources = '`Источники:`\n\n' + sources
            st.info(formatted_short_answer)
            st.info(formatted_sources)

            st.session_state.messages.append({"role": "assistant", "content": formatted_short_answer})
            st.session_state.messages.append({"role": "assistant", "content": formatted_sources})
    except Exception as e:
        print(e)
        formatted_answer = '`Ответ:`\n\n' + "Проблема с поиском, попробуйте позже."
        st_answer.info(formatted_answer)

        st.session_state.messages.append({"role": "assistant", "content": formatted_answer})






