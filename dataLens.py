import os
import sys
import json
import openai
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set OpenAI API Key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is not set!", file=sys.stderr)
    sys.exit(1)

# Read input from JSON file or environment variable
try:
    if "GPTSCRIPT_INPUT" in os.environ:
        input_json = os.getenv("GPTSCRIPT_INPUT")
    else:
        with open("query.json", "r") as file:
            input_json = file.read()
    
    params = json.loads(input_json)
except FileNotFoundError:
    print("Error: query.json file not found!", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON input! {str(e)}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error reading input: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Extract and validate parameters
nlp_query = params.get("nlp_query")
spreadsheet_path = params.get("spreadsheet_path")
visualization = params.get("visualization", None)  # This will be None unless explicitly requested

if not nlp_query:
    print("Error: Missing 'nlp_query' parameter!", file=sys.stderr)
    sys.exit(1)
if not spreadsheet_path:
    print("Error: Missing 'spreadsheet_path' parameter!", file=sys.stderr)
    sys.exit(1)

# Validate spreadsheet path
if not Path(spreadsheet_path).exists():
    print(f"Error: Spreadsheet file '{spreadsheet_path}' not found!", file=sys.stderr)
    sys.exit(1)

# Load spreadsheet into DuckDB
try:
    file_ext = Path(spreadsheet_path).suffix.lower()
    if file_ext == '.csv':
        df = pd.read_csv(spreadsheet_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(spreadsheet_path)
    else:
        print(f"Error: Unsupported file format '{file_ext}'!", file=sys.stderr)
        sys.exit(1)
    
    # Create connection and register the dataframe
    conn = duckdb.connect(database=':memory:')
    conn.register('df_view', df)
    conn.execute("CREATE TABLE df AS SELECT * FROM df_view")
except Exception as e:
    print(f"Error loading data: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Extract Table Schema
try:
    table_schema = conn.execute("PRAGMA table_info(df)").fetchdf().to_string(index=False)
except Exception as e:
    print(f"Error extracting schema: {str(e)}", file=sys.stderr)
    sys.exit(1)

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

def generate_sql(nlp_text, table_schema):
    """Generate SQL from natural language using OpenAI API"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
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
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Lower temperature for more deterministic SQL generation
        )
        
        raw_sql = response.choices[0].message.content.strip()
        return clean_sql_query(raw_sql)
    except openai.APIError as e:
        print(f"OpenAI API Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating SQL: {str(e)}", file=sys.stderr)
        sys.exit(1)

# Generate SQL from NLP query
sql_query = generate_sql(nlp_query, table_schema)
print("Generated SQL Query:\n", sql_query)  # Debugging output

# Execute SQL Query
try:
    result_df = conn.execute(sql_query).fetchdf()
    if result_df.empty:
        print("Warning: Query returned no results")
    result_json = result_df.to_dict(orient="records")
except Exception as e:
    print(f"SQL Execution Error: {str(e)}", file=sys.stderr)
    print(f"Attempted query: {sql_query}", file=sys.stderr)
    sys.exit(1)

# Generate Visualization ONLY if explicitly requested
image_path = None
if visualization:  # Only proceed if visualization parameter was provided
    valid_viz_types = ["bar", "line", "scatter", "hist", "pie"]
    
    # Validate visualization type
    if visualization not in valid_viz_types:
        print(f"Warning: '{visualization}' is not a supported visualization type. Using 'bar'.", file=sys.stderr)
        visualization = "bar"  # Default to bar if invalid type provided
    
    # Only create visualization if we have enough data
    if len(result_df.columns) >= 2 and not result_df.empty:
        try:
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
            
            image_path = "chart.png"
            plt.savefig(image_path, dpi=300)
            plt.close()
        except Exception as e:
            print(f"Visualization Error: {str(e)}", file=sys.stderr)
            # Continue execution even if visualization fails
    else:
        print("Warning: Not enough data for visualization", file=sys.stderr)

# Clean up DuckDB resources
try:
    conn.execute("DROP TABLE IF EXISTS df")
    conn.unregister('df_view')
    conn.close()
except:
    pass  # Ignore cleanup errors

# Return Final Output
output = {
    "nlp_query": nlp_query,
    "generated_sql": sql_query,
    "data_preview": result_json[:10],  # Limit preview to 10 records
    "row_count": len(result_df) if 'result_df' in locals() else 0
}

# Only include chart_path in the output if visualization was requested and successful
if image_path:
    output["chart_path"] = image_path

print(json.dumps(output, indent=4))