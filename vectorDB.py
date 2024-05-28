from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

from runbooks.web_pages.web_data_sources import runbook_web_sources

import os
import re


def initialize_agent_executor(st, llm, prompt):
    """
    This initializes all tools for multiple data sources and
    creates an agent executor for the Nebula Runbook LLM.
    """

    message = st.empty()
    message.text("Initializing Agents. Please wait.....")
    
    # Initialize the text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)
    embeddings= OpenAIEmbeddings()

    if "agent_executor" not in st.session_state:
        tools = []

        tools.extend(_embed_pdf_runbooks(text_splitter=text_splitter,
                                         embeddings=embeddings))
        
        tools.extend(_embed_web_page_runbooks(text_splitter=text_splitter,
                                              embeddings=embeddings))

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


def _embed_pdf_runbooks(text_splitter, embeddings, override=False):
    """
    This reads all pdf runbook documents from a directory and creates 
    retriever tools for the pdf runbooks.
    """

    # @TODO: Store Vector Database in a more permanent location
    pdf_retriever_tools = []
    
    for filename in os.listdir("./runbooks/pdfs"):
        if not filename.endswith('.pdf'):
            continue

        index_path = f"./vector_db/pdf_{filename}.faiss"

        # Load the PDF
        pdf_loader = PyPDFLoader("./runbooks/pdfs/" + filename)
        doc = pdf_loader.load()
        final_documents = text_splitter.split_documents(doc)

        if os.path.exists(index_path) and not override:
            vector = FAISS.load_local(index_path, 
                                      embeddings,
                                      allow_dangerous_deserialization=True)
        else:
            # Create a vector from the documents
            vector = FAISS.from_documents(final_documents,
                                        embeddings)
            # Save the vector to disk
            vector.save_local(index_path)

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


def _embed_web_page_runbooks(text_splitter, embeddings, override=False):
    """
    This reads given webpages and ingests the data
    into the vector database.
    """
    web_tools = []

    for source in runbook_web_sources:
        index_path = f"./vector_db/web_{source['name']}.faiss"

        if os.path.exists(index_path) and not override:
            web_vector = FAISS.load_local(index_path, 
                                          embeddings,
                                          allow_dangerous_deserialization=True)
        
        else:
            loader = WebBaseLoader(source["url"])
            web_docs = loader.load()
            documents = text_splitter.split_documents(web_docs)
            web_vector = FAISS.from_documents(documents, embeddings)

            # Save the vector to disk
            web_vector.save_local(index_path)

        web_retriever = web_vector.as_retriever()

        name = source["name"]
        description = source["description"]
        web_retriever_tool = create_retriever_tool(web_retriever,
                                                   name,
                                                   description)
        
        web_tools.append(web_retriever_tool)
    
    return web_tools
    
