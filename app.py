import streamlit as st
import os

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv


load_dotenv()

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def create_vector_database():
    """
    This reads all runbook documents from a directory and creates a 
    vector database.
    """

    # Intialize the vector database only if not already in session
    # @TODO: Store Vector Database in a more permanent location
    if "vectors" not in st.session_state:
        message = st.empty()
        message.text("Vectorizing Runbooks. Please wait.....")
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./runbooks")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
                                                        st.session_state.embeddings)
        message.text("Database Initialized")
        
st.title("Nebula Runbook LLM")
st.subheader("A Runbook Assistant for Site Reliability Engineers")

# llm = Ollama(model="llama3")
llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_template(
    """
    You are a runbook assistant who helps in troubleshooting issues
    for Site Reliability Engineers.
    Please provide the most accurate response based on the
    question with steps and code where appropriate.
    <context>
    {context}
    </context>
    Question: {input}"""
)

input_prompt = st.text_input("What system issue are you facing?")

if st.button("Initialize Runbooks"):
    create_vector_database()
    

if input_prompt:
    with st.spinner("Thinking..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"input": input_prompt})
        st.write(response["answer"])

        with st.expander("Referrences:"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("------------------------------------------")

