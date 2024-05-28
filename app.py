import streamlit as st
import os

# from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from vectorDB import initialize_agent_executor

from dotenv import load_dotenv


load_dotenv()

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("Nebula Runbook LLM")
st.subheader("A Runbook Assistant for Site Reliability Engineers")

# llm = Ollama(model="llama3")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

prompt =  ChatPromptTemplate.from_messages(
        [
            ("system", """You are a runbook assistant who helps in troubleshooting issues
                          for Site Reliability Engineers.
                          Please provide the most accurate response that includes 
                          troubleshooting and steps to fix based on the
                          question with steps. Ensure that your response is accuracte
                          and matches the context. You will be tipped $1000 if correct."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

# Initialize the agent executor if not already initialized
if st.button("Initialize Runbooks"):
    initialize_agent_executor(st, llm, prompt)

input_prompt = st.text_input("What system issue are you facing?")

if input_prompt:
    if not st.session_state.agent_executor:
        st.write("Please initialize the runbooks first")
    else:     
        with st.spinner("Thinking..."):
            response = st.session_state.agent_executor.invoke({"input": input_prompt})
            st.write(response["output"])

            ## @TODO: To find how to get context for agent executor
            # with st.expander("Referrences:"):
            #     for i, doc in enumerate(response["context"]):
            #         st.write(doc.page_content)
            #         st.write("------------------------------------------")

