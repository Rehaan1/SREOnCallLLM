from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

import os
import re


def initialize_agent_executor(st, llm, prompt):

    message = st.empty()
    message.text("Initializing Agents. Please wait.....")
    if "agent_executor" not in st.session_state:
        tools = []

        tools.extend(_embed_pdf_runbooks())

        # Create the agent
        agent = create_openai_tools_agent(llm, 
                                        tools, 
                                        prompt)
            
        # Store the agent executor in the session state
        st.session_state.agent_executor = AgentExecutor(agent=agent, 
                                                        tools=tools, 
                                                        verbose=True)
        
        message.text("Vector Database and Agent Initialized")
    else:
        message.text("Vector Database and Agent already initialized")   


def _embed_pdf_runbooks():
    """
    This reads all pdf runbook documents from a directory and creates 
    retriever tools for the pdf runbooks.
    """

    # @TODO: Store Vector Database in a more permanent location
    pdf_retriever_tools = []
    
    # Initialize the text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)
    embeddings= OpenAIEmbeddings()
    for filename in os.listdir("./runbooks"):
        if not filename.endswith('.pdf'):
            continue
        # Load the PDF
        pdf_loader = PyPDFLoader("./runbooks/" + filename)
        doc = pdf_loader.load()
        
        # Split the PDf based on chunks
        final_documents = text_splitter.split_documents(doc)
        # Create a vector from the documents
        vector = FAISS.from_documents(final_documents,
                                      embeddings)
        # Create a retriever from the vector
        retriever = vector.as_retriever()
        
        # Remove any special characters and file extensions from file name
        name = re.sub(r'\W+', ' ', filename)
        # Fill all white spaces with underscores
        name = re.sub(r'\s+', '_', name)
        # Choose the first chunk as the description
        description = final_documents[0].page_content
        description = re.sub(r'\W+', ' ', description)
        
        # Create the retriever tool
        retriever_tool = create_retriever_tool(retriever,
                                               name,
                                               description)
        pdf_retriever_tools.append(retriever_tool)
    
    return pdf_retriever_tools
