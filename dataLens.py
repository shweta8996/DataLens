import os
import sys
import json
import openai
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import asyncio
from io import BytesIO, StringIO

# Define supported file types
SUPPORTED_SPREADSHEET_TYPES = (".csv", ".xlsx", ".xls")

def clean_sql_query(sql_text):
    """
    Cleans the SQL query by removing any Markdown formatting like ```sql ``` blocks.
    """
    sql_text = sql_text.strip()
    
    # Remove markdown code block markers
    lines = sql_text.split('\n')
    if lines and (lines[0].startswith('```') or lines[0].strip() == 'sql'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    
    return '\n'.join(lines).strip()

async def main():
    # Read query parameters
    try:
        input_json = sys.argv[1] if len(sys.argv) > 1 else ""
        params = json.loads(input_json)
    except json.JSONDecodeError as e:
        sys.exit(f"Error: Invalid JSON input! {str(e)}")
    
    # Extract and validate parameters
    nlp_query = params.get("nlp_query")
    spreadsheet_path = params.get("spreadsheet_path")
    visualization = params.get("visualization", None)
    openai_api_key = params.get("openai_api_key")
    
    if not nlp_query:
        sys.exit("Error: Missing 'nlp_query' parameter!")
    if not spreadsheet_path:
        sys.exit("Error: Missing 'spreadsheet_path' parameter!")
    if not openai_api_key:
        sys.exit("Error: Missing 'openai_api_key' parameter!")
    
    if not Path(spreadsheet_path).suffix.lower() in SUPPORTED_SPREADSHEET_TYPES:
        sys.exit(f"Error: Unsupported file format '{spreadsheet_path}'")
    
    # Load spreadsheet
    try:
        if spreadsheet_path.endswith('.csv'):
            df = pd.read_csv(spreadsheet_path)
        else:
            df = pd.read_excel(spreadsheet_path)
        
        conn = duckdb.connect(database=':memory:')
        conn.register('df_view', df)
        conn.execute("CREATE TABLE df AS SELECT * FROM df_view")
    except Exception as e:
        sys.exit(f"Error loading data: {str(e)}")
    
    # Extract Table Schema
    try:
        table_schema = conn.execute("PRAGMA table_info(df)").fetchdf().to_string(index=False)
    except Exception as e:
        sys.exit(f"Error extracting schema: {str(e)}")
    
    # Generate SQL Query using OpenAI
    try:
        prompt = f"""
        Convert the following request into a valid SQL query for a DuckDB table:
        - The SQL MUST always include 'FROM df'.
        - The SQL MUST always include all necessary columns used in ORDER BY.
        - The SQL MUST follow proper DuckDB syntax.
        - The table schema is:
        {table_schema}
        
        Request: "{nlp_query}"
        """
    

        client = openai.Client(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        sql_query = clean_sql_query(response.choices[0].message.content.strip())

    except Exception as e:
        sys.exit(f"Error generating SQL: {str(e)}")
    
    # Execute SQL Query
    try:
        result_df = conn.execute(sql_query).fetchdf()
    except Exception as e:
        sys.exit(f"SQL Execution Error: {str(e)}")
    
    # Generate Visualization (if requested)
    if visualization and not result_df.empty and len(result_df.columns) >= 2:
        plt.figure(figsize=(10, 6))
        x_col = result_df.columns[0]
        y_col = result_df.select_dtypes(include=['number']).columns.tolist()[0] if len(result_df.select_dtypes(include=['number']).columns) > 0 else result_df.columns[1]
        if visualization == "bar":
            result_df.plot(kind="bar", x=x_col, y=y_col, ax=plt.gca())
        elif visualization == "line":
            result_df.plot(kind="line", x=x_col, y=y_col, ax=plt.gca())
        elif visualization == "scatter":
            result_df.plot(kind="scatter", x=x_col, y=y_col, ax=plt.gca())
        elif visualization == "hist":
            result_df[y_col].plot(kind="hist", ax=plt.gca())
        elif visualization == "pie" and len(result_df) <= 10:
            result_df.plot(kind="pie", y=y_col, labels=result_df[x_col], ax=plt.gca())
        plt.title(f"{y_col} by {x_col}")
        plt.tight_layout()
        plt.savefig("visualization.png")
        plt.close()
    
    # Return Output
    output = {
        "nlp_query": nlp_query,
        "generated_sql": sql_query,
        "data_preview": result_df.head(10).to_dict(orient="records"),
        "row_count": len(result_df)
    }
    if visualization:
        output["chart_path"] = "visualization.png"
    
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    asyncio.run(main())