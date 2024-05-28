import streamlit as st
import os

# from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from vectorDB import create_vector_database

from dotenv import load_dotenv


load_dotenv()

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


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

# The retriever is initialized only once
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if st.button("Initialize Runbooks"):
    create_vector_database(st)

input_prompt = st.text_input("What system issue are you facing?")


if input_prompt:
    if not st.session_state.retriever:
        st.write("Please initialize the runbooks first")
    else:     
        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)

            response = retrieval_chain.invoke({"input": input_prompt})
            st.write(response["answer"])

            with st.expander("Referrences:"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("------------------------------------------")

