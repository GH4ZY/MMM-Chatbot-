import os
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from chroma_db import vectorstore
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated
import tempfile
import os
#Imports the retriever we built in chroma_db.py.
#Brings in the SQL database (mmm.db) and Streamlit tools.


custom_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are an AI assistant specialized in Marketing Mix Modeling with access to an MMM SQL database and additional tools. Use them both.
Answer the user's query exactly as he wants. 

Use the following tools:
{tools}

When given a question, follow this format:

Question: {input}
Thought: Think about what to do.
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: The final answer to the original question

Begin!

{agent_scratchpad}
"""
)


#You return a retriever that can semantically find the top 5 relevant chunks for a user query.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAzoSiVqamIZZ6aAhO28m5KzpK3p8o1zj8" # Replace with your actual API key

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp-image-generation", temperature=0.5)

# Initialize the QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#vector_tool: Use PDFs to explain variables
vector_tool = Tool(
    name="DbVariablesMeaning",
    func=qa_chain.run,
    description="Provides explanations about database schema and variables meanings."
)


# Define the path to your SQLite database
dbfilepath = (Path(__file__).parent / "mmm.db").absolute()
creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
db = SQLDatabase(create_engine("sqlite:///", creator=creator))

# Initialize the SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_tools = toolkit.get_tools()


repl = PythonREPL()
#Executes matplotlib code and returns a chart image (in base64).
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Execute Python code and return path to generated plot image."""
    try:
        # Create a temporary file to save the plot
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            # Modify the code to save the plot to the temporary file
            modified_code = f"""
import matplotlib.pyplot as plt
{code}
plt.savefig(r'{tmpfile.name}')
"""
            exec(modified_code, {})
            return tmpfile.name
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"


# Create the SQL agent
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    extra_tools=[vector_tool, python_repl_tool],
    agent_type="tool-calling"
)

# Streamlit application
st.set_page_config(page_title="SQL Chat Agent", layout="wide")
st.title("💬 Chat with Your SQLite Database")

# Input field for user query
user_query = st.text_input("Enter your question about the database:")

# Display the database path
st.write(f"Using database at: {dbfilepath}")

# Initialize Streamlit callback handler
st_callback = StreamlitCallbackHandler(st.container())

# Execute the agent when the user submits a query
# Execute the agent when the user submits a query
if user_query:
    with st.spinner("Processing your query..."):
        try:
            response = agent.run(user_query, callbacks=[st_callback])
            if os.path.exists(response):
                st.image(response)
                os.remove(response)  # Clean up the temporary file
            else:
                st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
