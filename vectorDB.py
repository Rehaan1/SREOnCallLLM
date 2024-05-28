from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

def create_vector_database(st):
    """
    This reads all runbook documents from a directory and creates a 
    vector database.
    """

    message = st.empty()
    # Intialize the vector database only if not already in session
    # @TODO: Store Vector Database in a more permanent location
    if "vectors" not in st.session_state:
        message.text("Vectorizing Runbooks. Please wait.....")
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./runbooks")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
                                                        st.session_state.embeddings)
        st.session_state.retriever = st.session_state.vectors.as_retriever()
        message.text("Database Initialized")
    else:
        message.text("Database already initialized")