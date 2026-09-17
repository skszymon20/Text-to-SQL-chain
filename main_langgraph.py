import os
from IPython.display import Image, display
from typing import TypedDict, Optional
from IPython.display import Image, display
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import re
import urllib.request


load_dotenv()
# download database if not there
db_filename = "chinook.db"
if not os.path.exists(db_filename):
    print(f"📥 {db_filename} not found. Downloading the official Chinook dataset...")
    url = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
    urllib.request.urlretrieve(url, db_filename)
    print("✅ Download complete!")
db = SQLDatabase.from_uri(f"sqlite:///{db_filename}")
db_schema = db.get_table_info()

# Configure Gemini Model (temperature=0 for strict syntax accuracy)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

# ==========================================
# 1. DEFINE STATE AND PROMPTS
# ==========================================

class State(TypedDict):
    question: str
    sql_query: Optional[str]
    query_result: Optional[str]
    error_log: Optional[str]
    retry_count: int

# System prompt giving Gemini the DB schema structure
sql_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert SQL analyst. Given a user question and a database schema, write a syntactically correct SQLite query. Return ONLY the raw SQL string without any markdown code blocks, backticks, or dialogue.\n\nDatabase Schema:\n{schema}"),
    ("user", "Question: {question}\n\nPrevious Failed Attempt Error (if any): {error_log}\nGenerate the SQL query:")
])

explanation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an analytics assistant. Translate raw database queries into a natural language response."),
    ("user", "User Question: {question}\nExecuted SQL: {sql_query}\nRaw Query Result: {query_result}\nProvide a friendly, direct answer:")
])

# ==========================================
# 2. DEFINE THE COMPUTATIONAL NODES
# ==========================================

def generate_sql_node(state: State) -> dict:
    """Generates SQL using the schema and any past execution error logs."""
    print(f"🤖 [Node: Generate] Attempting SQL generation (Retry count: {state['retry_count']})")
    
    formatted_prompt = sql_generation_prompt.format(
        schema=db_schema, 
        question=state["question"], 
        error_log=state.get("error_log") or "None"
    )
    
    response = llm.invoke(formatted_prompt)
    assert response.content, "LLM returned an empty response. Check your model configuration and prompt."
    assert 'text' in response.content[0], "LLM response does not contain expected 'text' field."
    resp = response.content[0]['text']
    clean_query = resp.strip().replace("```sql", "").replace("```", "")
    
    return {"sql_query": clean_query}

def execute_sql_node(state: State) -> dict:
    """Attempts running the query. Catches syntax exceptions to pass back to state."""
    print(f"⚙️ [Node: Execute] Running: {state['sql_query']}")
    try:
        # Run standard SQLAlchemy abstraction query execution
        result = db.run(state["sql_query"])
        return {"query_result": result, "error_log": None}
    except Exception as e:
        print(f"❌ [Node: Execute] Hit Error: {str(e)}")
        # Catch and store database runtime exception string
        return {"error_log": str(e), "retry_count": state["retry_count"] + 1}

def explain_result_node(state: State) -> dict:
    """Takes database array outputs and turns it into natural insights."""
    print("✍️ [Node: Explain] Synthesizing output data...")
    formatted_prompt = explanation_prompt.format(
        question=state["question"],
        sql_query=state["sql_query"],
        query_result=state["query_result"]
    )
    response = llm.invoke(formatted_prompt)
    assert response.content, "LLM returned an empty response. Check your model configuration and prompt."
    assert 'text' in response.content[0], "LLM response does not contain expected 'text' field."
    resp = response.content[0]['text']
    print(f"\n💡 Final Response:\n{resp}\n")
    return {}

# ==========================================
# 3. DEFINE CONDITIONAL ROUTING ROUTINES
# ==========================================

def check_execution_status(state: State):
    """Determines whether to retry, output results, or fail gracefully."""
    if state.get("error_log") is None:
        return "explain"
    elif state["retry_count"] >= 3:
        print("💥 [Edge] Reached maximum retry ceiling. Aborting workflow loops.")
        return "explain"
    else:
        print("🔄 [Edge] Rerouting execution flow back to healing node...")
        return "generate"

# ==========================================
# 4. ORCHESTRATE AND COMPILE THE GRAPH
# ==========================================

workflow = StateGraph(State)

# Add processing nodes
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("explain_result", explain_result_node)

# Set workflow entry points
workflow.set_entry_point("generate_sql")

# Link logical edges
workflow.add_edge("generate_sql", "execute_sql")

# Bind self-healing dynamic retry branch conditions
workflow.add_conditional_edges(
    "execute_sql",
    check_execution_status,
    {
        "generate": "generate_sql",   # Loops back to self-heal
        "explain": "explain_result"   # Proceeds forward to wrap up
    }
)

workflow.add_edge("explain_result", END)

# Compile into our runtime interface app
app = workflow.compile()

# save the graph structure
try:
    with open("sql_self_healing_graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    print("🎉 Success! The graph has been saved as 'sql_self_healing_with_guardrails_graph.png' in your project directory.")
except Exception as e:
    print(f"Could not save graph image: {e}")

# initial_input = {
#     "question": "Show me the top 3 longest tracks.",
#     "retry_count": 0,
#     "sql_query": None,
#     "query_result": None,
#     "error_log": None
# }
# app.invoke(initial_input)

initial_input = {
    "question": "Which employee managed the most revenue from customers in the USA, and what was their total sales number?",
    "retry_count": 0,
    "sql_query": None,
    "query_result": None,
    "error_log": None
}
app.invoke(initial_input)