#!/usr/bin/env python3
"""
Integrated UI Chat Client
Intelligent dialogue client integrating Openterface AI Chat Client and UI-Ins model functionality
"""

from __future__ import annotations

import requests
import json
import os
import socket
import datetime
import re
import logging
from typing import Dict, Any, Optional, List
from PIL import Image, ImageDraw
from dotenv import load_dotenv

# Configure logging to file
log_file = "debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables early to determine execution modes
load_dotenv()
LLM_MODE = os.getenv("LLM_MODE", "online").lower()
UI_INS_MODE = os.getenv("UI_INS_MODE", "online").lower()
RAG_MODE = os.getenv("RAG_MODE", "online").lower()

# Conditional imports based on execution modes
# LlamaIndex modules needed for local RAG mode
if RAG_MODE == "local":
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
        from llama_index.core.readers.base import BaseReader
        from llama_index.core.schema import Document
        from llama_index.embeddings.openai import OpenAIEmbedding
        RAG_AVAILABLE = True
    except ImportError:
        print("⚠️  RAG local mode dependencies not found.")
        print("    Install with: pip install -r requirements_ops_cli.txt")
        RAG_AVAILABLE = False
else:
    RAG_AVAILABLE = False

# Common imports
import quopri
from bs4 import BeautifulSoup


# Define MHTMLReader only if RAG is available in local mode
if RAG_AVAILABLE:
    class MHTMLReader(BaseReader):
        """Custom MHTML file reader"""
        def load_data(self, file_path: str, extra_info: dict = None) -> list[Document]:
            """Load MHTML file and extract visible content"""
            with open(file_path, 'rb') as f:
                mhtml_content = f.read()
            
            # Decode to string
            mhtml_content = mhtml_content.decode('utf-8', errors='ignore')
            
            # Find HTML part
            boundary_match = re.search(r'boundary="([^"]+)"', mhtml_content)
            if not boundary_match:
                return [Document(text="", extra_info=extra_info or {})]
            
            boundary = boundary_match.group(1)
            parts = mhtml_content.split(f'--{boundary}')
            
            html_content = ""
            for part in parts:
                if 'Content-Type: text/html' in part:
                    part_content = re.sub(r'Content-Type: text/html[\s\S]*?Content-Location: [^\n]*\n', '', part)
                    part_content = part_content.strip()
                    if part_content:
                        html_content = part_content
                        break
            
            if not html_content:
                return [Document(text="", extra_info=extra_info or {})]
            
            # Decode quoted-printable encoding
            html_content = quopri.decodestring(html_content).decode('utf-8', errors='ignore')
            html_content = html_content.replace('\r\n', '\n')
            
            # Extract visible content
            soup = BeautifulSoup(html_content, 'lxml')
            for script in soup(['script', 'style', 'noscript', 'meta', 'link', 'head']):
                script.extract()
            
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n+', '\n', text)
            
            return [Document(text=text, extra_info=extra_info or {})]

# Global variables for storing conversation history
conversation_history = []
is_multiturn_mode = False

# Global variables for multilingual support
current_translations = {}
current_language = "en"

# UI-Ins server configuration (loaded from .env if available)
UI_INS_API_URL = os.getenv("UI_INS_API_URL", "http://localhost:2345/v1/chat/completions")

# RAG configuration (always define, but only used when RAG_AVAILABLE is True)
RAG_API_BASE = os.getenv("RAG_API_BASE", "http://localhost:11434/v1")
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "qwen3-embedding:0.6b")
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "./index")
RAG_DOCS_DIR = os.getenv("RAG_DOCS_DIR", "./docs")

# Global variables for RAG functionality (only available when RAG_AVAILABLE is True)
if RAG_AVAILABLE:
    index = None
    retriever = None
    rag_enabled = False
else:
    # In online mode, RAG is not available
    rag_enabled = False


def load_translations(lang_code: str = "en") -> Dict[str, Any]:
    """
    Load translation file for specified language
    """
    try:
        lang_file = os.path.join("i18n", f"{lang_code}.json")
        with open(lang_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # If specified language file doesn't exist, load default language (English)
        with open(os.path.join("i18n", "en.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(_("errors.translation_load_failed", error=str(e)))
        return {}


def _(key: str, **kwargs) -> str:
    """
    Translation function supporting formatted strings
    """
    global current_translations
    # Support nested keys like "messages.connecting"
    keys = key.split(".")
    value = current_translations
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return key  # Return original key if not found
    
    # Support formatted strings
    if kwargs and isinstance(value, str):
        return value.format(**kwargs)
    return value


def switch_language(lang_code: str) -> bool:
    """
    Switch language
    """
    global current_translations, current_language
    if lang_code in ["zh", "en"]:
        new_translations = load_translations(lang_code)
        if new_translations:
            current_translations = new_translations
            current_language = lang_code
            print(_("messages.lang_switched", lang=lang_code))
            return True
    return False


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string
    """
    import base64
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return f"Image encoding error: {str(e)}"


def setup_llamaindex():
    """
    Set up LlamaIndex environment (only available when RAG_MODE is local)
    """
    if not RAG_AVAILABLE:
        raise Exception("LlamaIndex is not available when RAG_MODE is online")
    
    from llama_index.core import Settings
    Settings.embed_model = OpenAIEmbedding(
        model_name=RAG_EMBED_MODEL,
        api_base=RAG_API_BASE,
    )


def build_index_from_docs(docs_dir: str = RAG_DOCS_DIR, index_dir: str = RAG_INDEX_DIR) -> bool:
    """
    Build index from document directory (only available when RAG_MODE is local)
    """
    global is_multiturn_mode
    
    if not RAG_AVAILABLE:
        print("⚠️  RAG functionality is only available when RAG_MODE is set to 'local'")
        return False
    
    # Check if in multiturn conversation mode
    if is_multiturn_mode:
        print(_("rag.multiturn_restricted"))
        return False
    
    try:
        print(_("rag.loading_docs", dir=docs_dir))
        
        # Set up LlamaIndex environment
        setup_llamaindex()
        
        # Configure file extractor
        file_extractor = {
            ".mhtml": MHTMLReader()
        }
        
        # Read documents
        documents = SimpleDirectoryReader(docs_dir, file_extractor=file_extractor).load_data()
        print(_("rag.docs_loaded", count=len(documents)))
        
        # Build index
        print(_("rag.index_building"))
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True,
        )
        
        # Save index
        print(_("rag.saving_index", dir=index_dir))
        index.storage_context.persist(persist_dir=index_dir)
        
        print(_("rag.index_built"))
        return True
    except Exception as e:
        print(f"❌ Index build failed: {str(e)}")
        return False


def load_index(index_dir: str = RAG_INDEX_DIR) -> "VectorStoreIndex":
    """
    Load index from directory (only available when RAG_MODE is local)
    """
    global is_multiturn_mode
    
    if not RAG_AVAILABLE:
        raise Exception("LlamaIndex is not available when RAG_MODE is online")
    
    # Check if in multiturn conversation mode
    if is_multiturn_mode:
        print(_("rag.multiturn_restricted"))
        raise Exception(_("rag.multiturn_restricted"))
    
    # Set up LlamaIndex environment
    setup_llamaindex()
    
    # Load index
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    return load_index_from_storage(storage_context)


def retrieve_relevant_docs(query: str, top_k: int = 3) -> List[str]:
    """
    Retrieve relevant documents from index based on query (only available when RAG_MODE is local)
    """
    global index, retriever, is_multiturn_mode
    
    if not RAG_AVAILABLE:
        return []
    
    # Check if in multiturn conversation mode
    if is_multiturn_mode:
        print(_("rag.multiturn_restricted"))
        return []
    
    # If index not loaded, try to load from directory
    if index is None:
        try:
            index = load_index()
            retriever = index.as_retriever(similarity_top_k=top_k)
        except Exception as e:
            print(f"❌ Failed to load index: {str(e)}")
            return []
    
    # Perform retrieval
    try:
        print(f"🔍 Retrieving documents related to query '{query}'...")
        query_results = retriever.retrieve(query)
        
        # Format retrieval results
        retrieved_content = []
        for i, result in enumerate(query_results, 1):
            node = result.node
            score = result.score
            content = node.text.strip()
            
            print(f"   📄 Result {i}: Similarity {score:.4f}")
            print(f"      File name: {node.metadata.get('file_name', 'N/A')}")
            print(f"      Page number: {node.metadata.get('page_label', 'N/A')}")
            print(f"      Content: {content[:100]}...")
            
            retrieved_content.append(content)
        
        return retrieved_content
    except Exception as e:
        print(f"❌ Document retrieval failed: {str(e)}")
        return []


def test_api_connection(api_url: str) -> bool:
    """
    Test if API connection is working by making a simple chat request
    """
    try:
        logger.info(f"Testing API connection to: {api_url}")
        
        # First, try to get model list (for APIs that support it)
        models_endpoint = api_url.replace("/chat/completions", "/models")
        try:
            test_response = requests.get(models_endpoint, timeout=5)
            logger.debug(f"Models endpoint test status: {test_response.status_code}")
            if test_response.status_code == 200:
                logger.info("API connection successful (models endpoint)")
                return True
        except Exception as e:
            logger.debug(f"Models endpoint not available: {str(e)}")
        
        # If models endpoint doesn't work, try a simple chat completion request
        # This is more reliable for APIs like Dashscope
        llm_api_key = os.getenv("LLM_API_KEY", "EMPTY")
        test_payload = {
            "messages": [
                {"role": "user", "content": "ping"}
            ],
            "max_tokens": 10,
            "model": os.getenv("MODEL", "default")
        }
        
        logger.debug(f"Sending test request to {api_url}")
        test_response = requests.post(
            api_url,
            json=test_payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {llm_api_key}"
            },
            timeout=5
        )
        
        logger.info(f"API test response status: {test_response.status_code}")
        logger.debug(f"API test response: {test_response.text}")
        
        if test_response.status_code == 200:
            logger.info("API connection successful (chat endpoint)")
            return True
        elif test_response.status_code == 401:
            logger.error("API authentication failed - check LLM_API_KEY")
            return False
        elif test_response.status_code == 404:
            logger.error("API endpoint not found - check API_URL and MODEL")
            return False
        else:
            logger.warning(f"API returned status {test_response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("API connection test timed out")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Cannot connect to API: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"API connection test failed: {str(e)}", exc_info=True)
        return False


def get_api_response(prompt: str, api_url: str = "http://localhost:11434/v1/chat/completions", model: str = "default", image_path: str = None, history: list = None, retrieved_docs: List[str] = None) -> str:
    """
    Send request to local OpenAI-compatible API and get response
    """
    global is_multiturn_mode
    
    try:
        # If there are retrieved documents, add them to the prompt
        enhanced_prompt = prompt
        if retrieved_docs and len(retrieved_docs) > 0:
            # Check if in multiturn conversation mode
            if is_multiturn_mode:
                return "RAG functionality (loading documents) is not available in multiturn conversation mode. Please switch to single-turn mode and try again."
            
            retrieved_content = "\n".join([f"[Relevant Document {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)])
            enhanced_prompt = f"Answer the user's question based on the following relevant documents:\n{retrieved_content}\n\nUser question: {prompt}"
        
        # Construct request body - support different API formats
        if "/chat" in api_url or "chat" in api_url.lower():
            # Chat API format
            messages = []
            
            # If there's conversation history and in multiturn mode, add history
            if history is not None and len(history) > 0:
                messages.extend(history)
            
            # Add current user message
            if image_path and os.path.exists(image_path):
                # Add image content to message
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(image_path)}"}}
                    ]
                })
                messages.append({
                    "role": "system",
                    "content": "Use the same language as user types. Enclose the name of the element (not the coordinate) with <click></click>, which is the element to click in the next step."
                })
            else:
                messages.append({
                    "role": "user",
                    "content": enhanced_prompt
                })
            
            payload = {
                "messages": messages,
                "max_tokens": 4096,
                "model": model
            }
        else:
            # Completions API format
            payload = {
                "prompt": enhanced_prompt,
                "max_tokens": 4096,
                "model": model
            }
        
        # Get API key from environment variable, use "EMPTY" as default if not found
        llm_api_key = os.getenv("LLM_API_KEY", "EMPTY")
        
        # Log request details
        logger.debug(f"API URL: {api_url}")
        logger.debug(f"Model: {model}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        logger.debug(f"Headers: Content-Type: application/json, Authorization: Bearer [hidden]")
        
        # Send POST request
        response = requests.post(
            api_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {llm_api_key}"
            },
            timeout=120
        )
        
        # Log response details
        logger.info(f"API Response Status: {response.status_code}")
        logger.debug(f"Response Headers: {dict(response.headers)}")
        logger.debug(f"Response Body: {response.text}")
        
        # Check response status
        if response.status_code == 200:
            data = response.json()
            # Extract response content based on different API return formats
            if "choices" in data and len(data["choices"]) > 0:
                # For OpenAI-compatible APIs
                choice = data["choices"][0]
                if "text" in choice:
                    return choice["text"].strip()
                elif "message" in choice:
                    # For Chat interface
                    return choice["message"]["content"].strip()
                elif "delta" in choice:
                    # For streaming responses
                    return choice["delta"].get("content", "").strip()
                else:
                    return json.dumps(choice, indent=2, ensure_ascii=False)
            else:
                return _("api_errors.no_response")
        else:
            logger.error(f"API request failed with status {response.status_code}")
            logger.error(f"Response text: {response.text}")
            return _("api_errors.status_error", code=response.status_code)
            
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return _("api_errors.timeout")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        return _("api_errors.connection_error")
    except Exception as e:
        logger.error(f"Unexpected error in get_api_response: {str(e)}", exc_info=True)
        return _("api_errors.error_occurred", error=str(e))


def get_last_image_from_server(host: str = 'localhost', port: int = 12345, output_dir: str = './images', timeout: int = 120) -> str:
    """
    Get latest image from Openterface server
    """
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Log function
        def log(message):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
        
        log(_("image_server.connecting"))
        log(_("image_server.target", host=host, port=port))
        
        # Create TCP connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(timeout)  # Set timeout
        
        # Connect to server
        client_socket.connect((host, port))
        log(_("image_server.connect_success", host=host, port=port))
        
        # Send "lastimage" command
        command = "lastimage"
        client_socket.send(command.encode('utf-8'))
        log(_("image_server.sent_command", command=command))
        
        # Receive response
        response = b""
        total_received = 0
        expected_image_size = 0
        
        # First read header information
        while True:
            try:
                data = client_socket.recv(4096)
                if not data:
                    break
                response += data
                total_received += len(data)
                
                # If received image data, try to parse header
                if response.startswith(b"IMAGE:") and b'\n' in response:
                    # Parse image size
                    header_end = response.find(b'\n')
                    if header_end != -1:
                        image_size_str = response[6:header_end].decode('utf-8')
                        expected_image_size = int(image_size_str)
                        break
                elif b"ERROR:" in response or b"STATUS:" in response:
                    # If error or status response, stop receiving immediately
                    break
            except socket.timeout:
                log(_("image_server.timeout"))
                break
            except Exception as e:
                log(_("image_server.receive_error", error=str(e)))
                break
        
        # Continue receiving remaining image data
        if expected_image_size > 0:
            expected_total = 6 + len(str(expected_image_size)) + 1 + expected_image_size  # Header + newline + image data size
            while len(response) < expected_total and total_received < expected_total + 10000:  # Add some tolerance
                try:
                    data = client_socket.recv(min(4096, expected_total - len(response)))
                    if not data:
                        break
                    response += data
                    total_received += len(data)
                except socket.timeout:
                    log(_("image_server.timeout"))
                    break
                except Exception as e:
                    log(_("image_server.receive_image_error", error=str(e)))
                    break
        
        log(_("image_server.received_data", size=len(response)))
        
        # Check if it's image data
        if response.startswith(b"IMAGE:"):
            # Parse image data
            try:
                # Separate image size info and actual image data
                header_end = response.find(b'\n')
                if header_end != -1:
                    image_size_str = response[6:header_end].decode('utf-8')
                    image_size = int(image_size_str)
                    image_data = response[header_end+1:]
                    
                    # Verify data integrity
                    if len(image_data) >= image_size:
                        # Generate filename
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"last_image_{timestamp}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        # Save image to file
                        with open(filepath, 'wb') as f:
                            f.write(image_data[:image_size])
                        
                        log(_("image_server.image_saved", path=filepath))
                        log(_("image_server.image_size", size=image_size))
                        
                        # Show success message
                        print(_("messages.image_success"))
                        print(_("messages.file_path", path=filepath))
                        print(_("messages.file_size", size=image_size))
                        print(_("messages.save_time", time=timestamp))
                        
                        # Close connection
                        client_socket.close()
                        return filepath
                        
            except ValueError as ve:
                log(_("image_server.image_data_error", error=str(ve)))
            except Exception as e:
                log(_("image_server.process_image_error", error=str(e)))
        
        # Check if it's error response
        elif response.startswith(b"ERROR:"):
            error_msg = response.decode('utf-8')[6:].strip()
            log(_("image_server.server_error", error=error_msg))
            client_socket.close()
            return f"❌ Server error: {error_msg}"
            
        # Check if it's status response
        elif response.startswith(b"STATUS:"):
            status_msg = response.decode('utf-8')[7:].strip()
            log(_("image_server.server_status", status=status_msg))
            client_socket.close()
            return f"📈 Server status: {status_msg}"
            
        else:
            # Unknown response format
            log(_("image_server.unknown_response", response=response[:100]))
            # Try to show text content
            try:
                text_content = response.decode('utf-8', errors='ignore')
                if text_content.strip():
                    log(_("image_server.text_content", content=text_content[:200]))
            except:
                pass
        
        # Close connection
        client_socket.close()
        log(_("image_server.connection_closed"))
        return _("image_server.image_failed")
        
    except socket.timeout:
        log(_("image_server.timeout_error"))
        return _("image_server.timeout_error")
    except ConnectionRefusedError:
        log(_("image_server.refused_error"))
        return _("image_server.refused_error")
    except socket.gaierror as e:
        log(_("image_server.dns_error", error=str(e)))
        return _("image_server.dns_error", error=str(e))
    except Exception as e:
        log(_("image_server.unknown_error", error=str(e)))
        return _("image_server.unknown_error", error=str(e))


def extract_click_content(text: str) -> Optional[str]:
    """
    Extract content from <click> tags in text
    
    Args:
        text: Text containing <click> tags
        
    Returns:
        Extracted tag content, or None if not found
    """
    pattern = r'<click>(.*?)</click>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None


def draw_rectangle(image_path, top_left, bottom_right, output_path=None):
    """
    Draw rectangle on specified image
    """
    # Open image
    img = Image.open(image_path)
    
    # Create drawing object
    draw = ImageDraw.Draw(img)
    
    # Draw rectangle
    draw.rectangle([top_left, bottom_right], outline="red", width=2)
    
    # Save image
    if output_path is None:
        output_path = image_path
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    img.save(output_path)
    print(f"Rectangle drawn to: {output_path}")
    return output_path


def parse_coordinates(raw_string: str) -> tuple[int, int]:
    """
    Parse coordinates from UI-Ins API response
    Supported format: Element found at coordinates [x,y]
    """
    # Match number sequences inside square brackets
    matches = re.findall(r'\[([^\]]+)\]', raw_string)
    
    for match in matches:
        # Split numbers and convert to integers
        numbers = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
        
        # If at least 2 numbers found, return first two as coordinate points
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
    
    return -1, -1


def call_ui_ins_api(image_path: str, instruction: str, ui_ins_api_url: str, ui_ins_model: str) -> str:
    """
    Call UI-Ins server API for element localization
    
    Args:
        image_path: Image path
        instruction: Localization instruction
        ui_ins_api_url: UI-Ins API endpoint
        ui_ins_model: UI-Ins model name
        
    Returns:
        API response result
    """
    print(f"\nCalling UI-Ins server for element localization: {instruction}")
    
    # Prepare request data
    messages = [
        {
            "role":"system",
            "content": "Provide the coordinate of the element in the screenshot. The coordinate should be in the format of [x, y], enclosed in square brackets."
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(image_path)}"}},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    
    payload = {
        "messages": messages,
        "max_tokens": 128,
        "model": ui_ins_model
    }

    ui_ins_api_key = os.getenv("UI_INS_API_KEY", "EMPTY")
    
    # Log request details
    logger.debug(f"UI-Ins API URL: {ui_ins_api_url}")
    logger.debug(f"UI-Ins Model: {ui_ins_model}")
    logger.debug(f"UI-Ins Instruction: {instruction}")
    logger.debug(f"UI-Ins Payload (image base64 truncated): {json.dumps({**payload, 'messages': [{**msg, 'content': '[truncated]' if isinstance(msg.get('content'), list) else msg.get('content')} for msg in payload.get('messages', [])]}, indent=2, ensure_ascii=False)}")
    
    # Send request
    try:
        response = requests.post(
            ui_ins_api_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ui_ins_api_key}"
            },
            timeout=120
        )
        
        logger.info(f"UI-Ins API Response Status: {response.status_code}")
        logger.debug(f"UI-Ins Response Headers: {dict(response.headers)}")
        logger.debug(f"UI-Ins Response Body: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                result = data["choices"][0]["message"]["content"].strip()
                logger.info(f"UI-Ins API success: {result}")
                return result
            else:
                logger.warning("UI-Ins API: No response choices found")
                return "UI-Ins API: No response choices found"
        else:
            logger.error(f"UI-Ins API Error: {response.status_code} - {response.text}")
            return f"UI-Ins API Error: {response.status_code} - {response.text}"
    except Exception as e:
        logger.error(f"UI-Ins API Connection Error: {str(e)}", exc_info=True)
        return f"UI-Ins API Connection Error: {str(e)}"


def process_ui_element_request(image_path: str, instruction: str, ui_ins_api_url: str, ui_ins_model: str):
    """
    Process UI element localization request via API
    """
    # Call UI-Ins server API
    response = call_ui_ins_api(image_path, instruction, ui_ins_api_url, ui_ins_model)
    
    # Display response result
    print(f"\nUI-Ins server response: {response}")
    
    # Parse coordinates
    point_x, point_y = parse_coordinates(response)
    
    if point_x != -1:
        print(f"Element localization successful, coordinates: ({point_x}, {point_y})")
        
        # Generate output image path
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join("./output", f"{name}_ui_ins{ext}")
        
        # Draw rectangle
        draw_rectangle(image_path, (point_x-20, point_y-20), (point_x+20, point_y+20), output_path)
        
        print(f"\n✅ Element localization completed:")
        print(f"   Localized element: {instruction}")
        print(f"   Coordinates: [{point_x}, {point_y}]")
        print(f"   Result image: {output_path}")
    else:
        print(f"❌ Coordinate parsing failed: {response}")


def print_header():
    """
    Print program title and help information
    """
    print("=" * 60)
    print(_("app_title"))
    print(_("app_description"))
    print("=" * 60)
    print("💡 " + _("feature_list.title"))
    print(_("feature_list.real_time_chat"))
    print(_("feature_list.multiple_api"))
    print(_("feature_list.friendly_ui"))
    print(_("feature_list.error_handling"))
    print(_("feature_list.ui_element_locator"))
    print("=" * 60)
    
    # Print execution modes
    print("\n📋 Execution Modes:")
    print(f"   LLM Mode: {LLM_MODE.upper()} {'(Using remote API)' if LLM_MODE == 'online' else '(Local model)'}")
    print(f"   UI-INS Mode: {UI_INS_MODE.upper()} {'(Using remote API)' if UI_INS_MODE == 'online' else '(Local model)'}")
    print(f"   RAG Mode: {RAG_MODE.upper()} {'(Disabled)' if RAG_MODE == 'online' else '(Local index available)'}")
    print("=" * 60)


def print_help():
    """
    Print help information
    """
    print(f"\n{_("help_title")}")
    print(_("commands.ask_question"))
    print(_("commands.quit"))
    print(_("commands.clear"))
    print(_("commands.help"))
    print(_("commands.info"))
    print(_("commands.model"))
    print(_("commands.image"))
    print(_("commands.load_docs"))
    print(_("commands.multiturn"))
    print(_("commands.single"))
    print(_("commands.lang_help"))
    print(_("commands.lang_switch"))
    print("=" * 60)


def print_api_info(api_url: str):
    """
    Print API information
    """
    print(f"\n{_("api_info.title")}")
    print(_("api_info.address", api_url=api_url))
    connection_status = "✅ Available" if test_api_connection(api_url) else "⚠️  Unavailable"
    print(_("api_info.status", status=connection_status))


def main():
    """
    Main function: Provide interactive conversation interface
    """
    import sys
    
    # Initialize translation
    global current_translations, current_language
    
    # Check command line arguments, support --lang option
    for i, arg in enumerate(sys.argv):
        if arg in ['--lang', '-l'] and i + 1 < len(sys.argv):
            lang_code = sys.argv[i + 1].lower()
            if lang_code in ['en', 'zh']:
                current_language = lang_code
    
    current_translations = load_translations(current_language)
    
    # Check if need to automatically show help and exit
    if '--help' in sys.argv or '-h' in sys.argv:
        print_header()
        print_help()
        return
    
    print_header()
    
    # Environment variables are already loaded at module startup
    # Load configuration from .env if explicitly set
    api_url_configured = "API_URL" in os.environ
    model_configured = "MODEL" in os.environ
    ui_ins_api_url_configured = "UI_INS_API_URL" in os.environ
    ui_ins_model_configured = "UI_INS_MODEL" in os.environ
    
    # Get API configuration
    api_url = os.getenv("API_URL", "http://localhost:11434/v1/chat/completions")
    model = os.getenv("MODEL", "qwen3-vl:32b")
    
    # Get UI-Ins configuration
    ui_ins_api_url = os.getenv("UI_INS_API_URL", "http://localhost:2345/v1/chat/completions")
    ui_ins_model = os.getenv("UI_INS_MODEL", "ui-ins-7b")
    
    # Only prompt for API configuration if not configured in .env
    if not api_url_configured or not model_configured:
        print(f"\n{_("messages.config_api")}")
        
        if not api_url_configured:
            print(_("messages.default_address", api_url=api_url))
            custom_api = input(_("messages.enter_custom_api")).strip()
            if custom_api:
                api_url = custom_api
        else:
            print(f"   Using API URL from .env: {api_url}")
        
        if not model_configured:
            print(_("messages.default_model", model=model))
            custom_model = input(_("messages.enter_custom_model")).strip()
            if custom_model:
                model = custom_model
        else:
            print(f"   Using model from .env: {model}")
    else:
        print(f"\n✓ API Configuration loaded from .env")
        print(f"   API URL: {api_url}")
        print(f"   Model: {model}")
    
    # Test connection
    print(_("messages.connecting", api_url=api_url))
    if test_api_connection(api_url):
        print(_("messages.connection_success"))
    else:
        print(_("messages.connection_warning"))
        print(_("messages.connection_advice"))
    
    # Only prompt for UI-Ins configuration if not configured in .env
    if not ui_ins_api_url_configured or not ui_ins_model_configured:
        print(f"\n{_("messages.config_ui_ins")}")
        
        if not ui_ins_api_url_configured:
            print(_("messages.default_ui_ins_address", ui_ins_api_url=ui_ins_api_url))
            custom_ui_ins_api = input(_("messages.enter_custom_ui_ins_api")).strip()
            if custom_ui_ins_api:
                ui_ins_api_url = custom_ui_ins_api
        else:
            print(f"   Using UI-Ins API URL from .env: {ui_ins_api_url}")
        
        if not ui_ins_model_configured:
            print(_("messages.default_ui_ins_model", ui_ins_model=ui_ins_model))
            custom_ui_ins_model = input(_("messages.enter_custom_ui_ins_model")).strip()
            if custom_ui_ins_model:
                ui_ins_model = custom_ui_ins_model
        else:
            print(f"   Using UI-Ins model from .env: {ui_ins_model}")
    else:
        print(f"\n✓ UI-Ins Configuration loaded from .env")
        print(f"   UI-Ins API URL: {ui_ins_api_url}")
        print(f"   UI-Ins Model: {ui_ins_model}")
    
    # Test UI-Ins connection
    print(_("messages.connecting_ui_ins", ui_ins_api_url=ui_ins_api_url))
    if test_api_connection(ui_ins_api_url):
        print(_("messages.ui_ins_connection_success"))
    else:
        print(_("messages.ui_ins_connection_warning"))
        print(_("messages.ui_ins_connection_advice"))
    
    print("\n" + "-" * 60)
    print(_("messages.start_chat"))
    print("-" * 60)
    
    # Interactive conversation loop
    while True:
        try:
            # Get user input
            user_input = input(_("messages.your_question")).strip()
            
            # Handle various commands
            if user_input.lower() in ['/quit', '/exit', '/q']:
                print(_("messages.goodbye"))
                break
            
            # Handle clear command
            if user_input.lower() in ['/clear', '/cls']:
                print(_("messages.history_cleared"))
                # Clear conversation history
                global conversation_history
                conversation_history = []
                continue
            
            # Handle help command
            if user_input.lower() == '/help':
                print_help()
                continue
            
            # Handle info command
            if user_input.lower() == '/info':
                print_api_info(api_url)
                continue
            
            # Handle model switch command
            if user_input.lower() == '/model':
                new_model = input("   " + _("messages.enter_model") + " ").strip()
                if new_model:
                    model = new_model
                    print(_("messages.model_switched", model=model))
                continue
            
            # Handle language command
            if user_input.lower().startswith('/lang'):
                parts = user_input.split()
                if len(parts) == 1:
                    # Show current language
                    print(_("messages.current_lang", lang=current_language))
                elif len(parts) == 2:
                    # Switch language
                    lang_code = parts[1].lower()
                    if lang_code in ['en', 'zh']:
                        switch_language(lang_code)
                    else:
                        print(_("messages.lang_invalid"))
                else:
                    print(_("messages.lang_invalid"))
                continue
            
            # Handle multiturn conversation mode command
            if user_input.lower() == '/multiturn':
                print(_("messages.multiturn_mode"))
                print(_("messages.multiturn_info_1"))
                print(_("messages.multiturn_info_2"))
                print(_("messages.multiturn_info_3"))
                print(_("messages.multiturn_info_4"))
                print(_("messages.multiturn_info_5"))
                global is_multiturn_mode, index, retriever
                is_multiturn_mode = True
                # Clear loaded index, disable RAG functionality
                index = None
                retriever = None
                print(_("rag.rag_disabled"))
                continue
            
            # Handle exit multiturn conversation mode command
            if user_input.lower() == '/single':
                print(_("messages.single_mode"))
                is_multiturn_mode = False
                # Clear conversation history
                conversation_history = []
                continue
            
            # Handle "load docs" command
            if user_input.lower() == '/load docs':
                print(_("messages.loading_docs"))
                success = build_index_from_docs()
                if success:
                    global rag_enabled
                    rag_enabled = True
                    print(_("messages.docs_loaded"))
                else:
                    print(_("messages.docs_load_failed"))
                continue
            
            # Handle image command
            if user_input.lower() == '/image':
                # Get image from server
                print(_("messages.getting_image"))
                openterface_host = os.getenv("OPENTERFACE_HOST", "localhost")
                openterface_port = int(os.getenv("OPENTERFACE_PORT", "12345"))
                image_path = get_last_image_from_server(host=openterface_host, port=openterface_port)
                print(_("messages.server_response", response=image_path))
                if image_path and image_path.startswith("./images"):
                    print(_("messages.image_obtained", filename=os.path.basename(image_path)))
                    # Clearly prompt user to enter question
                    question = input(_("messages.enter_question")).strip()
                    if not question:
                        print(_("messages.question_empty"))
                        continue
                    # Send request with image and question
                    print(_("messages.processing"))
                    print(_("messages.please_wait"))
                    
                    # If RAG functionality is enabled, retrieve relevant documents
                    retrieved_docs = []
                    if rag_enabled:
                        retrieved_docs = retrieve_relevant_docs(question)
                    
                    # If in multiturn conversation mode, pass conversation history
                    if is_multiturn_mode:
                        response = get_api_response(question, api_url, model, image_path, conversation_history, retrieved_docs)
                    else:
                        response = get_api_response(question, api_url, model, image_path, retrieved_docs=retrieved_docs)
                    
                    # If in multiturn conversation mode, update conversation history
                    if is_multiturn_mode and response != "":
                        # Add user message to history (including image information)
                        conversation_history.append({
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(image_path)}"}}
                            ]
                        })
                        # Add AI response to history
                        conversation_history.append({"role": "assistant", "content": response})
                    
                    # Display response
                    print(_("messages.ai_response"))
                    print("-" * 40)
                    print(response)
                    print("-" * 40)
                    
                    # Check if response contains <click> tags
                    click_content = extract_click_content(response)
                    if click_content:
                        print(f"\n{_("ui_ins.detected_element_request", content=click_content)}")
                        process_ui_element_request(image_path, click_content, ui_ins_api_url, ui_ins_model)
                else:
                    if "❌" in image_path:
                        print(_("image_server.image_path_warning", image_path=image_path))
                        # Prompt user to enter other content again
                        continue
            else:
                # Handle empty input
                if not user_input:
                    print(_("messages.invalid_question"))
                    continue
                
                # Send request and display response
                print(_("messages.processing"))
                print(_("messages.please_wait"))
                
                # If RAG functionality is enabled, retrieve relevant documents
                retrieved_docs = []
                if rag_enabled:
                    retrieved_docs = retrieve_relevant_docs(user_input)
                
                # If in multiturn conversation mode, pass conversation history
                if is_multiturn_mode:
                    response = get_api_response(user_input, api_url, model, None, conversation_history, retrieved_docs)
                else:
                    response = get_api_response(user_input, api_url, model, retrieved_docs=retrieved_docs)
                
                # If in multiturn conversation mode, update conversation history
                if is_multiturn_mode and response != "":
                    # Add user message to history
                    conversation_history.append({"role": "user", "content": user_input})
                    # Add AI response to history
                    conversation_history.append({"role": "assistant", "content": response})
                
                # Display response
                print(_("messages.ai_response"))
                print("-" * 40)
                print(response)
                print("-" * 40)
            
        except KeyboardInterrupt:
            print(_("messages.interrupted"))
            break
        except Exception as e:
            print(_("messages.error_occurred", error=str(e)))
            print(_("messages.retry_advice"))


if __name__ == "__main__":
    main()