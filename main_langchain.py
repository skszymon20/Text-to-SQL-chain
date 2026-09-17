from langchain_community.utilities import SQLDatabase
import os
import urllib.request
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_sql_query_chain
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit


load_dotenv()

db_filename = "chinook.db"

# 1. Automatically download the real dataset if it's missing
if not os.path.exists(db_filename):
    print(f"📥 {db_filename} not found. Downloading the official Chinook dataset...")
    url = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
    urllib.request.urlretrieve(url, db_filename)
    print("✅ Download complete!")


# Connect to the SQLite local database file
db = SQLDatabase.from_uri(f"sqlite:///{db_filename}")

# Sanity check: ensure tables print out correctly
print("Available Tables:", db.get_usable_table_names())

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# This chain accepts natural language text and converts it strictly to a text SQL string
write_query_chain = create_sql_query_chain(llm, db)

# Run a test conversion pass
response = write_query_chain.invoke({"question": "How many employees are there in the company?"})
print("Generated SQL:", response)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(llm, toolkit=toolkit, verbose=True)

# Try a fully integrated analytical lookup
result = agent_executor.invoke({"input": "Who is the top-selling artist by revenue?"})
print("Final Insight:", result["output"])
