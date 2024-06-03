from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

import config
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

    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    tools = []
    
    # @TODO: Note: This needs to be updated to prevent 
    # Repetation of Code. This is only a demo code
    
    # PDF Tool
    pdf_vdb = Milvus(embedding_function=embeddings,
                      collection_name=config.VDB_PDF_COLLECTION,
                      connection_args={
                          "uri": config.ZILLIZ_CLOUD_URI,
                          "token": config.ZILLIZ_CLOUD_API_KEY,
                          "secure": True,
                      })
    
    pdf_retriver = pdf_vdb.as_retriever()

    # Note: This is just a demo. Use better name and description
    # as they are used in the LLM Reasoning agent
    pdf_tool = create_retriever_tool(pdf_retriver, 
                                     "pdf_sources_sre",
                                     "A compiled list of various common System issues")
    
    tools.append(pdf_tool)

    # Web Tool
    web_vdb = Milvus(embedding_function=embeddings,
                        collection_name=config.VDB_WEB_COLLECTION,
                        connection_args={
                            "uri": config.ZILLIZ_CLOUD_URI,
                            "token": config.ZILLIZ_CLOUD_API_KEY,
                            "secure": True,
                        })
    
    web_retriver = web_vdb.as_retriever()

    # Note: This is just a demo. Use better name and description
    # as they are used in the LLM Reasoning agent
    web_tool = create_retriever_tool(web_retriver,
                                        "web_sources_sre",
                                        "A compiled list of various common System issues")
    
    tools.append(web_tool)

   
    agent = create_openai_tools_agent(llm, tools, prompt)
    # Store the agent executor in the session state
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    message.text("Agents Initialized. Nebula Runbook Ready to use")


def check_data_source_embedding_exists(source, embedding, collection_name):
    """
    Checks if a data source is already embedded in the vector database based on
    source. Source can be pdf name or url.
    """
    vector_db = Milvus(embedding_function=embedding,
                       collection_name=collection_name,
                       connection_args={
                           "uri": config.ZILLIZ_CLOUD_URI,
                           "token": config.ZILLIZ_CLOUD_API_KEY,
                           "secure": True,
                       }, )
    expression = f"source LIKE '%{source}'"
    docs = vector_db.get_pks(expression)

    if docs is None or len(docs) == 0:
        return False

    return True


def _embed_pdf_runbooks(text_splitter, embeddings):
    """
    This reads all pdf runbook documents from a directory and creates 
    retriever tools for the pdf runbooks.
    """
    
    for filename in os.listdir("./runbooks/pdfs"):
        if not filename.endswith('.pdf'):
            continue

        if check_data_source_embedding_exists(filename, embeddings, config.VDB_PDF_COLLECTION):
            print("already embedded")
            continue

        # Load the PDF
        pdf_loader = PyPDFLoader("./runbooks/pdfs/" + filename)
        doc = pdf_loader.load()
        # Split the PDf based on chunks
        final_documents = text_splitter.split_documents(doc)
        
        Milvus.from_documents(final_documents,
                              embeddings,
                              collection_name=config.VDB_PDF_COLLECTION,
                              connection_args={
                                  "uri": config.ZILLIZ_CLOUD_URI,
                                  "token": config.ZILLIZ_CLOUD_API_KEY,
                                  "secure": True,
                              })

    return True


def vectorize_data():
    """
    This function vectorizes the data from the data sources.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)

    _embed_pdf_runbooks(text_splitter, embeddings)
    _embed_web_page_runbooks(text_splitter, embeddings)

    print("-----------Data Vectorized---------")



def _embed_web_page_runbooks(text_splitter, embeddings):
    """
    This reads given webpages and ingests the data
    into the vector database.
    """

    for source in runbook_web_sources:
        if check_data_source_embedding_exists(source["url"], embeddings, config.VDB_WEB_COLLECTION):
            print("already embedded")
            continue

        loader = WebBaseLoader(source["url"])
        web_docs = loader.load()
        documents = text_splitter.split_documents(web_docs)
        Milvus.from_documents(documents,
                              embeddings,
                              collection_name=config.VDB_WEB_COLLECTION,
                              connection_args={
                                  "uri": config.ZILLIZ_CLOUD_URI,
                                  "token": config.ZILLIZ_CLOUD_API_KEY,
                                  "secure": True,
                              })

    return True
