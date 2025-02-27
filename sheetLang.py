import os
import sys
import json
import openai
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

#  Set OpenAI API Key
import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is not set!", file=sys.stderr)
    sys.exit(1)

#  Read input from JSON file (or environment variable if available)
if "GPTSCRIPT_INPUT" in os.environ:
    input_json = os.getenv("GPTSCRIPT_INPUT")
else:
    with open("query.json", "r") as file:
        input_json = file.read()

try:
    params = json.loads(input_json)
except json.JSONDecodeError as e:
    print("Error: Invalid JSON input!", file=sys.stderr)
    sys.exit(1)

#  Extract parameters
nlp_query = params.get("nlp_query")
spreadsheet_path = params.get("spreadsheet_path")
visualization = params.get("visualization", None)

if not nlp_query or not spreadsheet_path:
    print("Error: Missing required inputs!", file=sys.stderr)
    sys.exit(1)

#  Load spreadsheet into DuckDB
df = pd.read_csv(spreadsheet_path)
duckdb.execute("CREATE TABLE df AS SELECT * FROM df")

#  Extract Table Schema
table_schema = duckdb.execute("DESCRIBE df").fetchdf().to_string(index=False)

def clean_sql_query(sql_text):
    """
    Cleans the SQL query by removing any Markdown formatting like ```sql ``` blocks.
    """
    sql_text = sql_text.strip()  # Remove leading/trailing spaces
    if sql_text.startswith("```"):  # If it starts with a Markdown block
        sql_text = sql_text.split("\n", 1)[-1]  # Remove the first line
    if sql_text.endswith("```"):
        sql_text = sql_text.rsplit("\n", 1)[0]  # Remove the last line
    return sql_text.strip()

def generate_sql(nlp_text, table_schema):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)  

    prompt = f"""
    Convert the following request into a **valid SQL query** for a DuckDB table:
    - The SQL **must** always include 'FROM df'.
    - The SQL **must** always include all necessary columns used in ORDER BY.
    - The table schema is:
    {table_schema}

    Request: "{nlp_text}"
    
    SQL Query (DO NOT include Markdown formatting, just return raw SQL):
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_sql = response.choices[0].message.content.strip()
    return clean_sql_query(raw_sql)



#  Generate SQL from NLP query
sql_query = generate_sql(nlp_query, table_schema)
print("Generated SQL Query:\n", sql_query)  # Debugging output

#  Execute SQL Query
try:
    result_df = duckdb.execute(sql_query).fetchdf()
    result_json = result_df.to_dict(orient="records")
except Exception as e:
    print("SQL Execution Error:", e, file=sys.stderr)
    sys.exit(1)

#  Generate Visualization (if requested)
image_path = None
if visualization:
    plt.figure(figsize=(6, 4))
    if visualization == "bar":
        result_df.plot(kind="bar", x=result_df.columns[0], y=result_df.columns[1])
    elif visualization == "line":
        result_df.plot(kind="line", x=result_df.columns[0], y=result_df.columns[1])

    image_path = "chart.png"
    plt.savefig(image_path)
    plt.close()

#  Return Final Output
output = {
    "nlp_query": nlp_query,
    "generated_sql": sql_query,
    "data_preview": result_json,
    "chart_path": image_path
}

print(json.dumps(output, indent=4))
