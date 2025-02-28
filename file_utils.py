import os
import sys
import json
import openai
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from io import BytesIO, StringIO
from file_utils import setup_logger, load_from_gptscript_workspace, save_to_gptscript_workspace

# Define supported file types
SUPPORTED_SPREADSHEET_TYPES = (".csv", ".xlsx", ".xls")

# Set up logger
logger = setup_logger("SQL_Spreadsheet_Assistant")

def clean_sql_query(sql_text):
    """
    Cleans the SQL query by removing any Markdown formatting like ```sql ``` blocks.
    """
    sql_text = sql_text.strip()
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
        logger.error(f"Error: Invalid JSON input! {str(e)}")
        sys.exit(1)
    
    # Extract and validate parameters
    nlp_query = params.get("nlp_query")
    spreadsheet_path = params.get("spreadsheet_path")
    visualization = params.get("visualization", None)
    openai_api_key = params.get("openai_api_key")
    
    if not nlp_query:
        logger.error("Error: Missing 'nlp_query' parameter!")
        sys.exit(1)
    if not spreadsheet_path:
        logger.error("Error: Missing 'spreadsheet_path' parameter!")
        sys.exit(1)
    if not openai_api_key:
        logger.error("Error: Missing 'openai_api_key' parameter!")
        sys.exit(1)
    
    file_ext = os.path.splitext(spreadsheet_path)[-1].lower()
    if file_ext not in SUPPORTED_SPREADSHEET_TYPES:
        logger.error(f"Error: Unsupported file format '{file_ext}'")
        sys.exit(1)
    
    # Load spreadsheet from GPTScript workspace
    try:
        logger.info(f"Loading file: {spreadsheet_path}")
        file_content = await load_from_gptscript_workspace(spreadsheet_path)
        if file_ext == '.csv':
            df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        else:
            df = pd.read_excel(BytesIO(file_content))
        
        conn = duckdb.connect(database=':memory:')
        conn.register('df_view', df)
        conn.execute("CREATE TABLE df AS SELECT * FROM df_view")
        logger.info(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)
    
    # Extract Table Schema
    try:
        table_schema = conn.execute("PRAGMA table_info(df)").fetchdf().to_string(index=False)
        logger.info("Schema extracted successfully")
    except Exception as e:
        logger.error(f"Error extracting schema: {str(e)}")
        sys.exit(1)
    
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
        
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        sql_query = clean_sql_query(response["choices"][0]["message"]["content"].strip())
        logger.info(f"Generated SQL Query: {sql_query}")
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        sys.exit(1)
    
    # Execute SQL Query
    try:
        result_df = conn.execute(sql_query).fetchdf()
        if result_df.empty:
            logger.warning("Query returned no results")
    except Exception as e:
        logger.error(f"SQL Execution Error: {str(e)}")
        sys.exit(1)
    
    # Save results to GPTScript workspace
    output_file = "query_results.json"
    output_content = json.dumps(result_df.to_dict(orient="records"), indent=4)
    await save_to_gptscript_workspace(output_file, output_content.encode("utf-8"))
    logger.info(f"Query results saved to {output_file}")
    
    # Clean up DuckDB resources
    conn.close()
    
    print(json.dumps({
        "nlp_query": nlp_query,
        "generated_sql": sql_query,
        "data_preview": result_df.head(10).to_dict(orient="records"),
        "row_count": len(result_df),
        "results_file": output_file
    }, indent=4))

if __name__ == "__main__":
    asyncio.run(main())
