import os
import sys
import logging
import gptscript
import io

def setup_logger(name):
    """Setup a logger that writes to sys.stderr. This will show in GPTScript's debugging logs.
    
    Args:
        name (str): The name of the logger.
    Returns:
        logging.Logger: The logger.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logging level
    # Create a stream handler that writes to sys.stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    # Create a log formatter
    formatter = logging.Formatter(
        "[NLP SQL Tool Debugging Log]: %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stderr_handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(stderr_handler)
    return logger

def prepend_base_path(base_path: str, file_path: str):
    """
    Prepend a base path to a file path if it's not already rooted in the base path.
    
    Args:
        base_path (str): The base path to prepend.
        file_path (str): The file path to check and modify.
    Returns:
        str: The modified file path with the base path prepended if necessary.
    """
    # Split the file path into parts for checking
    file_parts = os.path.normpath(file_path).split(os.sep)
    # Check if the base path is already at the root
    if file_parts[0] == base_path:
        return file_path
    # Prepend the base path
    return os.path.join(base_path, file_path)

async def save_to_gptscript_workspace(filepath: str, content) -> None:
    """Save content to GPTScript workspace.
    
    Args:
        filepath (str): Path to save the file to
        content: Content to save (string or bytes)
    """
    gptscript_client = gptscript.GPTScript()
    wksp_file_path = prepend_base_path("files", filepath)
    
    # Convert string content to bytes if needed
    if isinstance(content, str):
        content = content.encode("utf-8")
    
    await gptscript_client.write_file_in_workspace(wksp_file_path, content)

async def load_from_gptscript_workspace(filepath: str) -> bytes:
    """Load file content from GPTScript workspace.
    
    Args:
        filepath (str): Path to the file to load
    Returns:
        bytes: The file content as bytes
    """
    gptscript_client = gptscript.GPTScript()
    wksp_file_path = prepend_base_path("files", filepath)
    file_content = await gptscript_client.read_file_in_workspace(wksp_file_path)
    return file_content