from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

import os
import re

# def create_vector_database(st):
#     """
#     This reads all runbook documents from a directory and creates a 
#     vector database.
#     """

#     message = st.empty()
#     # Intialize the vector database only if not already in session
#     # @TODO: Store Vector Database in a more permanent location
#     if "vectors" not in st.session_state:
#         message.text("Vectorizing Runbooks. Please wait.....")
#         st.session_state.embeddings = OpenAIEmbeddings()
#         st.session_state.loader = PyPDFDirectoryLoader("./runbooks")
#         st.session_state.docs = st.session_state.loader.load()
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
#                                                                         chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
#                                                         st.session_state.embeddings)
#         st.session_state.retriever = st.session_state.vectors.as_retriever()
#         message.text("Database Initialized")
#     else:
#         message.text("Database already initialized")


def embed_pdf_runbooks(st, llm, prompt):
    """
    This reads all pdf runbook documents from a directory and creates 
    retriever tools for the pdf runbooks.
    """
    message = st.empty()

    # @TODO: Store Vector Database in a more permanent location
    if "pdf_retriever_tools" not in st.session_state:
        pdf_retriever_tools = []
        message.text("Embedding PDF Runbooks. Please wait.....")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=200)
        embeddings= OpenAIEmbeddings()

        for filename in os.listdir("./runbooks"):
            if not filename.endswith('.pdf'):
                continue
            pdf_loader = PyPDFLoader("./runbooks/" + filename)
            doc = pdf_loader.load()
            
            final_documents = text_splitter.split_documents(doc)
            vector = FAISS.from_documents(final_documents,
                                          embeddings)
            retriever = vector.as_retriever()
            
            # Remove any special characters and file extensions from file name
            name = re.sub(r'\W+', ' ', filename)
            # Fill all white spaces with underscores
            name = re.sub(r'\s+', '_', name)

            description = final_documents[0].page_content
            description = re.sub(r'\W+', ' ', description)
            
            retriever_tool = create_retriever_tool(retriever,
                                                   name,
                                                   description)
            pdf_retriever_tools.append(retriever_tool)
        
        st.session_state.pdf_retriever_tools = pdf_retriever_tools 
        agent = create_openai_tools_agent(llm, 
                                          pdf_retriever_tools, 
                                          prompt)
        st.session_state.agent_executor = AgentExecutor(agent=agent, 
                                                        tools=pdf_retriever_tools, 
                                                        verbose=False)   
        message.text("Vector Database Initialized")
        
    else:
        message.text("Vector Database already initialized")