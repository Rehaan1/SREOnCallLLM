import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
import config
from vectorDB import initialize_agent_executor


st.title("Nebula Runbook LLM")
st.subheader("A Runbook Assistant for Site Reliability Engineers")

# llm = Ollama(model="llama3")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=config.OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.SYSTEM_CONTEXT),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

# Initialize the agent executor if not already initialized
if st.button("Initialize Run books"):
    initialize_agent_executor(st, llm, prompt)

input_prompt = st.text_input("What system issue are you facing?")

if input_prompt:
    if "agent_executor" not in st.session_state or not st.session_state.agent_executor:
        st.write("Please initialize the run books first")
    else:     
        with st.spinner("Thinking..."):
            response = st.session_state.agent_executor.invoke({"input": input_prompt})
            st.write(response["output"])

            ## @TODO: To find how to get context for agent executor
            # with st.expander("Referrences:"):
            #     for i, doc in enumerate(response["context"]):
            #         st.write(doc.page_content)
            #         st.write("------------------------------------------")
