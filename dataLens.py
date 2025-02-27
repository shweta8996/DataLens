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

# Import helper utilities
from file_utils import load_from_gptscript_workspace, save_to_gptscript_workspace, setup_logger

# Define supported file types
SUPPORTED_SPREADSHEET_TYPES = (".csv", ".xlsx", ".xls")

# Set up logger
logger = setup_logger(__name__)

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

# Set model and API parameters
MODEL = os.getenv("OBOT_DEFAULT_LLM_MODEL", "gpt-4")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ["OPENAI_API_KEY"]

# Initialize OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
except Exception as e:
    sys.exit(f"ERROR: Failed to initialize OpenAI client: {e}")


def generate_sql(nlp_text, table_schema):
    """Generate SQL from natural language using OpenAI API"""
    try:
        prompt = f"""
        Convert the following request into a  valid SQL query for a DuckDB table:
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
            temperature=0.1  # Lower temperature for more deterministic SQL generation
        )
        
        raw_sql = response.choices[0].message.content.strip()
        return clean_sql_query(raw_sql)
    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        sys.exit(1)

async def main():
    # Get input file from environment variable
    input_file = os.getenv("INPUT_FILE", "")
    if not input_file:
        logger.error("Error: INPUT_FILE environment variable is not set")
        sys.exit(1)
    
    # Validate file type
    if not input_file.endswith(SUPPORTED_SPREADSHEET_TYPES):
        logger.error(f"Error: the input file must be one of: {SUPPORTED_SPREADSHEET_TYPES}")
        sys.exit(1)

    # Read query parameters
    try:
        input_json = os.getenv("GPTSCRIPT_INPUT", None)
        if not input_json:
            try:
                with open("query.json", "r") as file:
                    input_json = file.read()
            except FileNotFoundError:
                logger.warning("query.json not found, using default query")
                input_json = '{"nlp_query": "Summarize the data", "visualization": "bar"}'
        
        params = json.loads(input_json)
    except json.JSONDecodeError as e:
        logger.error(f"Error: Invalid JSON input! {str(e)}")
        sys.exit(1)
    
    # Extract and validate parameters
    nlp_query = params.get("nlp_query")
    visualization = params.get("visualization", None)
    
    if not nlp_query:
        logger.error("Error: Missing 'nlp_query' parameter!")
        sys.exit(1)

    # Load spreadsheet
    try:
        logger.info(f"Loading file: {input_file}")
        file_content = await load_from_gptscript_workspace(input_file)
        file_ext = Path(input_file).suffix.lower()
        
        # Load into pandas based on file type
        if file_ext == '.csv':
            # For CSV files, we need to decode the bytes
            df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(BytesIO(file_content))
        else:
            logger.error(f"Error: Unsupported file format '{file_ext}'!")
            sys.exit(1)
        
        # Create connection and register the dataframe
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
        logger.info(f"Schema extracted successfully")
    except Exception as e:
        logger.error(f"Error extracting schema: {str(e)}")
        sys.exit(1)

    # Generate SQL from NLP query
    logger.info(f"Generating SQL for query: {nlp_query}")
    sql_query = generate_sql(nlp_query, table_schema)
    logger.info(f"Generated SQL Query: {sql_query}")

    # Execute SQL Query
    try:
        result_df = conn.execute(sql_query).fetchdf()
        if result_df.empty:
            logger.warning("Query returned no results")
        result_json = result_df.to_dict(orient="records")
        logger.info(f"SQL executed successfully, returned {len(result_df)} rows")
    except Exception as e:
        logger.error(f"SQL Execution Error: {str(e)}")
        logger.error(f"Attempted query: {sql_query}")
        sys.exit(1)

    # Generate Visualization ONLY if explicitly requested
    image_path = None
    if visualization:  # Only proceed if visualization parameter was provided
        valid_viz_types = ["bar", "line", "scatter", "hist", "pie"]
        
        # Validate visualization type
        if visualization not in valid_viz_types:
            logger.warning(f"'{visualization}' is not a supported visualization type. Using 'bar'.")
            visualization = "bar"  # Default to bar if invalid type provided
        
        # Only create visualization if we have enough data
        if len(result_df.columns) >= 2 and not result_df.empty:
            try:
                logger.info(f"Creating {visualization} visualization")
                plt.figure(figsize=(10, 6))
                
                # Try to identify appropriate x and y columns
                x_col = result_df.columns[0]  # Default to first column for x-axis
                
                # For numeric columns, use the second column as y by default
                numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) > 0:
                    # If first column is numeric and there are other numeric columns, use the second numeric column
                    if x_col in numeric_cols and len(numeric_cols) > 1:
                        y_col = [col for col in numeric_cols if col != x_col][0]
                    # Otherwise use the first numeric column as y
                    else:
                        y_col = numeric_cols[0]
                        # If y_col is same as x_col and there are other columns, use a different x
                        if y_col == x_col and len(result_df.columns) > 1:
                            x_col = [col for col in result_df.columns if col != y_col][0]
                else:
                    # If no numeric columns, just use the second column
                    y_col = result_df.columns[1] if len(result_df.columns) > 1 else result_df.columns[0]
                
                # Create visualization based on type
                if visualization == "bar":
                    result_df.plot(kind="bar", x=x_col, y=y_col, ax=plt.gca())
                elif visualization == "line":
                    result_df.plot(kind="line", x=x_col, y=y_col, ax=plt.gca())
                elif visualization == "scatter":
                    result_df.plot(kind="scatter", x=x_col, y=y_col, ax=plt.gca())
                elif visualization == "hist":
                    result_df[y_col].plot(kind="hist", ax=plt.gca())
                elif visualization == "pie" and len(result_df) <= 10:  # Pie charts work best with few categories
                    result_df.plot(kind="pie", y=y_col, labels=result_df[x_col], ax=plt.gca())

                plt.title(f"{y_col} by {x_col}")
                plt.tight_layout()
                
                # Set up chart path based on input file name
                name, _ = os.path.splitext(os.path.basename(input_file))
                image_path = f"{name}_chart.png"
                
                # Save the figure to a BytesIO buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=300)
                buf.seek(0)
                
                # Save the visualization using the GPTScript utilities
                await save_to_gptscript_workspace(image_path, buf.getvalue())
                logger.info(f"Visualization saved to {image_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Visualization Error: {str(e)}")
                # Continue execution even if visualization fails
        else:
            logger.warning("Not enough data for visualization")

    # Clean up DuckDB resources
    try:
        conn.execute("DROP TABLE IF EXISTS df")
        conn.unregister('df_view')
        conn.close()
    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")

    # Return Final Output
    output = {
        "nlp_query": nlp_query,
        "generated_sql": sql_query,
        "data_preview": result_json[:10],  # Limit preview to 10 records
        "row_count": len(result_df) if 'result_df' in locals() else 0
    }

    # Include chart_path in the output if visualization was requested and successful
    if image_path:
        output["chart_path"] = image_path

    # Handle output file
    output_file = os.getenv("OUTPUT_FILE", "")
    if output_file.upper() == "NONE":
        print(json.dumps(output, indent=4))
    else:
        if output_file == "":
            name, _ = os.path.splitext(os.path.basename(input_file))
            output_file = f"{name}_analysis.json"
        
        try:
            logger.info(f"Saving analysis to {output_file}")
            await save_to_gptscript_workspace(output_file, json.dumps(output, indent=4))
            print(f"Analysis written to file: {output_file}")
        except Exception as e:
            print(json.dumps(output, indent=4))
            logger.error(f"Error writing to output file: {str(e)}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())