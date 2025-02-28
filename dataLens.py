import os
import sys
import json
import openai
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from io import BytesIO, StringIO
from pathlib import Path
from file_utils import setup_logger, load_from_gptscript_workspace, save_to_gptscript_workspace

# Define supported file types
SUPPORTED_SPREADSHEET_TYPES = (".csv", ".xlsx", ".xls")

# Set up logger
logger = setup_logger("SQL_Spreadsheet_Assistant")

# Set model and API parameters
MODEL = "gpt-4"
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

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

def prompt_for_api_key():
    """
    Prompt the user for their OpenAI API key if not provided.
    """
    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].strip():
        return os.environ["OPENAI_API_KEY"].strip()
    
    print("\n=== SQL Spreadsheet Assistant ===")
    print("\nThis tool requires an OpenAI API key to convert natural language to SQL.")
    api_key = input("\nPlease enter your OpenAI API key: ").strip()
    
    if not api_key:
        logger.error("Error: OpenAI API key is required.")
        sys.exit(1)
    
    print("\nThank you! Your API key will be used securely for this request only.\n")
    return api_key

def initialize_openai_client(api_key):
    """Initialize OpenAI client with provided API key"""
    try:
        from openai import OpenAI
        return OpenAI(base_url=BASE_URL, api_key=api_key)
    except Exception as e:
        sys.exit(f"ERROR: Failed to initialize OpenAI client: {e}")

def generate_sql(client, nlp_text, table_schema):
    """Generate SQL from natural language using OpenAI API"""
    try:
        prompt = f"""
        Convert the following request into a valid SQL query for a DuckDB table:
        - The SQL MUST always include 'FROM df'.
        - The SQL MUST always include all necessary columns used in ORDER BY.
        - The SQL MUST follow proper DuckDB syntax.
        - The table schema is:
        {table_schema}
        
        Request: "{nlp_text}"
        
        SQL Query (DO NOT include Markdown formatting, just return raw SQL):
        """
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return clean_sql_query(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        sys.exit(1)

async def main():
    # Get parameters from command line or GPTSCRIPT_INPUT
    try:
        if len(sys.argv) > 1:
            # Handle command line JSON input
            input_json = sys.argv[1]
        else:
            input_json = os.getenv("GPTSCRIPT_INPUT", "{}")
        
        # Parse JSON input
        try:
            params = json.loads(input_json)
        except json.JSONDecodeError as e:
            logger.error(f"Error: Invalid JSON input! {str(e)}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        sys.exit(1)
    
    # Extract and validate parameters
    nlp_query = params.get("nlp_query")
    visualization = params.get("visualization", None)
    spreadsheet_path = params.get("spreadsheet_path")
    openai_api_key = params.get("openai_api_key", "").strip() or prompt_for_api_key()

    # Validate required parameters
    if not nlp_query:
        logger.error("Error: Missing 'nlp_query' parameter!")
        sys.exit(1)
    if not spreadsheet_path:
        logger.error("Error: Missing 'spreadsheet_path' parameter!")
        sys.exit(1)

    # Initialize OpenAI client with provided API key
    client = initialize_openai_client(openai_api_key)
    
    # Validate file type
    if not Path(spreadsheet_path).suffix.lower() in SUPPORTED_SPREADSHEET_TYPES:
        logger.error(f"Error: Unsupported file format '{spreadsheet_path}'")
        sys.exit(1)

    # Load spreadsheet
    try:
        logger.info(f"Loading file: {spreadsheet_path}")
        file_ext = Path(spreadsheet_path).suffix.lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(spreadsheet_path)
        else:
            df = pd.read_excel(spreadsheet_path)
        
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

    # Generate and execute SQL Query
    try:
        sql_query = generate_sql(client, nlp_query, table_schema)
        logger.info(f"Generated SQL Query: {sql_query}")
        
        result_df = conn.execute(sql_query).fetchdf()
        if result_df.empty:
            logger.warning("Query returned no results")
        result_json = result_df.to_dict(orient="records")
        logger.info(f"SQL executed successfully, returned {len(result_df)} rows")
    except Exception as e:
        logger.error(f"SQL Execution Error: {str(e)}")
        sys.exit(1)

    # Generate Visualization if requested
    image_path = None
    if visualization:
        valid_viz_types = ["bar", "line", "scatter", "hist", "pie"]
        
        if visualization not in valid_viz_types:
            logger.warning(f"'{visualization}' is not a supported visualization type. Using 'bar'.")
            visualization = "bar"
        
        if len(result_df.columns) >= 2 and not result_df.empty:
            try:
                logger.info(f"Creating {visualization} visualization")
                plt.figure(figsize=(10, 6))
                
                # Identify appropriate x and y columns
                x_col = result_df.columns[0]
                numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) > 0:
                    if x_col in numeric_cols and len(numeric_cols) > 1:
                        y_col = [col for col in numeric_cols if col != x_col][0]
                    else:
                        y_col = numeric_cols[0]
                        if y_col == x_col and len(result_df.columns) > 1:
                            x_col = [col for col in result_df.columns if col != y_col][0]
                else:
                    y_col = result_df.columns[1] if len(result_df.columns) > 1 else result_df.columns[0]
                
                # Create visualization
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
                
                # Save visualization
                name = Path(spreadsheet_path).stem
                image_path = f"{name}_chart.png"
                plt.savefig(image_path, format='png', dpi=300)
                logger.info(f"Visualization saved to {image_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Visualization Error: {str(e)}")
        else:
            logger.warning("Not enough data for visualization")

    # Clean up DuckDB resources
    conn.execute("DROP TABLE IF EXISTS df")
    conn.unregister('df_view')
    conn.close()

    # Prepare output
    output = {
        "nlp_query": nlp_query,
        "generated_sql": sql_query,
        "data_preview": result_json[:10],
        "row_count": len(result_df)
    }

    if image_path:
        output["chart_path"] = image_path

    # Save and print results
    output_file = f"{Path(spreadsheet_path).stem}_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    asyncio.run(main()) 