#!/usr/bin/env python3
"""
UI-Ins Server
Standalone UI-Ins model server with OpenAI-compatible API
Supports both online (proxy) and local (inference) modes
"""

import requests
import json
import os
import datetime
import re
import sys
import logging
import argparse
import atexit
import signal
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Configure logging to file
log_file = "ui_ins_server.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables early to determine execution mode
load_dotenv()
UI_INS_MODE = os.getenv("UI_INS_MODE", "online").lower()

# Conditional imports based on execution mode
# torch and transformers only needed for local mode
if UI_INS_MODE == "local":
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from PIL import Image, ImageDraw
        UI_INS_AVAILABLE = True
    except ImportError:
        print("⚠️  UI-INS local mode dependencies not found.")
        print("    Install with: pip install -r requirements_ops_cli.txt")
        UI_INS_AVAILABLE = False
else:
    UI_INS_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# UI-Ins mode and configuration
logger.info(f"UI-INS Mode: {UI_INS_MODE}")

# UI-Ins model global variables (only used in local mode)
UI_INS_MODEL_PATH = os.getenv("UI_INS_MODEL_PATH", "D:\\AI\\models\\UI-Ins-7B")
ui_ins_model = None
ui_ins_processor = None
standby_mode = "cold"  # Default cold standby mode

# UI-Ins server configuration
UI_INS_SERVER_HOST = os.getenv("UI_INS_SERVER_HOST", "0.0.0.0")
UI_INS_SERVER_PORT = int(os.getenv("UI_INS_SERVER_PORT", "2345"))

# Remote UI-INS API configuration (for online mode)
UI_INS_REMOTE_API_URL = os.getenv("UI_INS_API_URL", "http://192.168.1.6:1234/v1/chat/completions")
UI_INS_REMOTE_API_KEY = os.getenv("UI_INS_API_KEY", "EMPTY")

app = Flask(__name__)


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint - shows server information
    """
    return jsonify({
        "name": "UI-Ins Server",
        "version": "1.0",
        "mode": UI_INS_MODE,
        "description": "OpenAI-compatible API for UI element localization",
        "endpoints": {
            "health": "GET /health",
            "models": "GET /v1/models",
            "chat": "POST /v1/chat/completions"
        },
        "documentation": {
            "health_check": "http://localhost:2345/health",
            "list_models": "http://localhost:2345/v1/models",
            "chat_completions": "POST http://localhost:2345/v1/chat/completions"
        }
    })


def parse_coordinates(raw_string: str) -> tuple[int, int]:
    """
    Parse coordinates from model response, supports multiple formats:
    - Single coordinate: [x,y]
    - Two coordinates: [x1,y1,x2,y2]
    - More coordinates: [x1,y1,x2,y2,x3,y3,...]
    When multiple coordinates are present, only the first coordinate is extracted
    """
    logger.debug(f"Parsing coordinates from: {raw_string}")
    # Match all sequences of numbers within square brackets
    matches = re.findall(r'\[([^\]]+)\]', raw_string)
    
    for match in matches:
        # Split numbers and convert to integers
        numbers = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
        
        # If there are at least 2 numbers, return the first two as the first coordinate point
        if len(numbers) >= 2:
            logger.debug(f"Coordinates parsed successfully: ({numbers[0]}, {numbers[1]})")
            return numbers[0], numbers[1]
    
    logger.warning(f"Failed to parse coordinates from: {raw_string}")
    return -1, -1





def load_ui_ins_model(model_path: str = UI_INS_MODEL_PATH):
    """
    Load UI-Ins model (only available in local mode)
    """
    if not UI_INS_AVAILABLE:
        raise Exception("UI-Ins local mode dependencies not available. Install with: pip install -r requirements_ops_cli.txt")
    
    global ui_ins_model, ui_ins_processor
    try:
        logger.info(f"Loading UI-Ins model from: {model_path}")
        print("Loading UI-Ins model...")
        ui_ins_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        ui_ins_processor = AutoProcessor.from_pretrained(model_path)
        logger.info("UI-Ins model loaded successfully")
        print("UI-Ins model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load UI-Ins model: {str(e)}", exc_info=True)
        raise


def run_ui_ins_inference(image_path: str, instruction: str) -> tuple[int, int]:
    """
    Run UI-Ins model inference to get coordinates
    
    Args:
        image_path: Path to the image
        instruction: Instruction text
        
    Returns:
        Coordinate point (x, y)
    """
    global ui_ins_model, ui_ins_processor
    
    try:
        logger.info(f"Starting inference for instruction: {instruction}")
        logger.debug(f"Image path: {image_path}")
        
        # Check if model is already loaded
        if ui_ins_model is None or ui_ins_processor is None:
            logger.info("Model not loaded, loading now...")
            load_ui_ins_model()
        
        # Load image
        logger.debug(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        logger.debug(f"Image loaded successfully, size: {image.size}")
        
        # Build messages
        messages = [
            {
                "role":"system",
                "content": "Provide the coordinate of the element in the screenshot. The coordinate should be in the format of [x, y], enclosed in square brackets."
            },
            {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
            }]
        
        logger.debug("Processing input for model inference")
        # Process input
        prompt = ui_ins_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = ui_ins_processor(text=[prompt], images=[image], return_tensors="pt").to(ui_ins_model.device)
        logger.debug("Input processed, starting model inference")
        
        # Generate response
        generated_ids = ui_ins_model.generate(**inputs, max_new_tokens=128)
        response_ids = generated_ids[0, len(inputs["input_ids"][0]):]
        raw_response = ui_ins_processor.decode(response_ids, skip_special_tokens=True)
        logger.info(f"Model inference completed, raw response: {raw_response}")
        print(f"\nRaw model response: {raw_response}")
        
        # Parse coordinates
        point_x, point_y = parse_coordinates(raw_response)
        logger.info(f"Inference result: coordinates ({point_x}, {point_y})")
        
        return point_x, point_y
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise


def cleanup_model():
    """
    Clean up model resources when program exits (local mode only)
    """
    if UI_INS_MODE != "local":
        logger.info("Online mode: no local model to cleanup")
        return
    
    global ui_ins_model, ui_ins_processor
    if ui_ins_model is not None or ui_ins_processor is not None:
        logger.info("Program exiting, cleaning up model resources...")
        print("Program exiting, cleaning up model resources...")
        if ui_ins_model is not None:
            ui_ins_model = None
        if ui_ins_processor is not None:
            ui_ins_processor = None
        # Force release GPU memory
        if UI_INS_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model resources cleaned up successfully")
        print("Model resources cleaned up successfully")


def unload_ui_ins_model():
    """
    Unload UI-Ins model and release resources (local mode only)
    """
    if UI_INS_MODE != "local":
        return
    
    global ui_ins_model, ui_ins_processor
    logger.info("Unloading UI-Ins model...")
    print("Unloading UI-Ins model...")
    if ui_ins_model is not None:
        # Clear model and release GPU memory
        ui_ins_model = None
    if ui_ins_processor is not None:
        ui_ins_processor = None
    # Force release GPU memory
    if UI_INS_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("UI-Ins model has been unloaded")
    print("UI-Ins model has been unloaded")


def process_ui_element_request(image_path: str, instruction: str) -> Dict[str, Any]:
    """
    Process UI element localization request (local mode only)
    """
    if not UI_INS_AVAILABLE:
        raise Exception("UI-Ins local mode not available")
    
    logger.info(f"Processing UI element request: {instruction}")
    print(f"\nLocating element: {instruction}")
    
    try:
        # Run UI-Ins model inference (will automatically load model if not loaded)
        point_x, point_y = run_ui_ins_inference(image_path, instruction)
        
        if point_x != -1:
            logger.info(f"Element located successfully at coordinates: ({point_x}, {point_y})")
            print(f"Element located successfully at coordinates: ({point_x}, {point_y})")
            
            return {
                "success": True,
                "coordinates": [point_x, point_y],
                "message": f"Element found at coordinates ({point_x}, {point_y})"
            }
        else:
            logger.warning("Failed to parse coordinates from model response")
            print("Failed to parse coordinates")
            return {
                "success": False,
                "coordinates": None,
                "message": "Failed to parse coordinates"
            }
    except Exception as e:
        logger.error(f"UI element request processing failed: {str(e)}", exc_info=True)
        raise


def proxy_to_remote_api(data: dict) -> Dict[str, Any]:
    """
    Proxy request to remote UI-INS API (online mode)
    """
    logger.info(f"Proxying request to remote API: {UI_INS_REMOTE_API_URL}")
    try:
        response = requests.post(
            UI_INS_REMOTE_API_URL,
            json=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {UI_INS_REMOTE_API_KEY}"
            },
            timeout=120
        )
        
        logger.info(f"Remote API response status: {response.status_code}")
        logger.debug(f"Remote API response: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Remote API error: {response.status_code}")
            return {
                "error": {
                    "message": f"Remote API returned {response.status_code}: {response.text}",
                    "type": "remote_api_error",
                    "param": None,
                    "code": None
                }
            }
    except Exception as e:
        logger.error(f"Proxy request failed: {str(e)}", exc_info=True)
        return {
            "error": {
                "message": f"Failed to connect to remote API: {str(e)}",
                "type": "proxy_error",
                "param": None,
                "code": None
            }
        }

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    OpenAI-compatible chat completions API
    Supports both local inference and remote API proxy
    """
    logger.info(f"Received chat completions API request (mode: {UI_INS_MODE})")
    
    # If in online mode, proxy to remote API
    if UI_INS_MODE == "online":
        logger.info("Online mode: proxying to remote API")
        data = request.json
        result = proxy_to_remote_api(data)
        
        if "error" in result:
            logger.error(f"Remote API error: {result}")
            return jsonify(result), 500
        else:
            return jsonify(result), 200
    
    # Local mode: process inference locally
    if not UI_INS_AVAILABLE:
        logger.error("Local mode requested but dependencies not available")
        return jsonify({
            "error": {
                "message": "UI-Ins local mode dependencies not available",
                "type": "invalid_request_error",
                "param": None,
                "code": None
            }
        }), 500
    
    try:
        data = request.json
        logger.debug(f"Request headers: {dict(request.headers)}")
        logger.debug(f"Request data keys: {list(data.keys()) if data else 'None'}")
        messages = data.get('messages', [])
        logger.debug(f"Number of messages: {len(messages)}")
        
        # Extract last user message
        user_message = None
        for msg in reversed(messages):
            if msg['role'] == 'user':
                user_message = msg
                break
        
        if not user_message:
            logger.warning("No user message found in request")
            return jsonify({
                "error": {
                    "message": "No user message found",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }), 400
        
        # Extract instruction and image
        instruction = None
        image_data = None
        
        logger.debug(f"Extracting content from user message")
        content = user_message.get('content', '')
        if isinstance(content, list):
            # Process multimodal content
            for item in content:
                if item['type'] == 'text':
                    instruction = item['text']
                elif item['type'] == 'image_url':
                    image_url = item['image_url']['url']
                    # Process base64 image data
                    if image_url.startswith('data:image/'):
                        import base64
                        import io
                        # Extract base64 data
                        base64_data = image_url.split(',')[1]
                        # Decode base64 data to bytes
                        image_bytes = base64.b64decode(base64_data)
                        # Create temporary image file
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        temp_image_path = f"./temp_images/ui_ins_temp_{timestamp}.jpg"
                        os.makedirs("./temp_images", exist_ok=True)
                        with open(temp_image_path, 'wb') as f:
                            f.write(image_bytes)
                        image_data = temp_image_path
        else:
            # Plain text content
            instruction = content
        
        if not instruction:
            logger.warning("No instruction found in request")
            return jsonify({
                "error": {
                    "message": "No instruction found",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }), 400
        
        logger.info(f"Instruction extracted: {instruction}")
        
        if not image_data:
            logger.warning("No image found in request")
            return jsonify({
                "error": {
                    "message": "No image found",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }), 400
        
        logger.info(f"Image extracted: {image_data}")
        
        # Process UI element localization request
        logger.info("Processing UI element localization request")
        result = process_ui_element_request(image_data, instruction)
        logger.info(f"Result: {result}")
        
        # Decide whether to unload model based on standby mode
        if standby_mode == "cold":
            # Cold standby mode: unload model after each request
            logger.debug("Cold standby mode: unloading model after request")
            unload_ui_ins_model()
        # Hot standby mode: keep model loaded
        
        # Generate response
        if result['success']:
            response_content = f"Element found at coordinates [{result['coordinates'][0]}, {result['coordinates'][1]}]"
        else:
            response_content = "Failed to locate element"
        
        logger.info(f"Sending response: {response_content}")
        return jsonify({
            "id": f"chatcmpl-{datetime.datetime.now().timestamp()}",
            "object": "chat.completion",
            "created": int(datetime.datetime.now().timestamp()),
            "model": "ui-ins-7b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
        
    except Exception as e:
        logger.error(f"API request failed: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        # Decide whether to unload model based on standby mode
        if standby_mode == "cold":
            # Cold standby mode: unload model even when exception occurs
            logger.debug("Cold standby mode: unloading model after error")
            unload_ui_ins_model()
        # Hot standby mode: keep model loaded
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_server_error",
                "param": None,
                "code": None
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def get_models():
    """
    Get list of available models
    """
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "ui-ins-7b",
                "object": "model",
                "created": int(datetime.datetime.now().timestamp()),
                "owned_by": "ui-ins",
                "root": "ui-ins-7b",
                "parent": None,
                "permission": []
            }
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "ok",
        "message": "UI-Ins Server is running",
        "mode": UI_INS_MODE,
        "model_loaded": ui_ins_model is not None if UI_INS_MODE == "local" else "N/A (online)",
        "standby_mode": standby_mode if UI_INS_MODE == "local" else "N/A (online)"
    })

def signal_handler(signum, frame):
    """Signal handler to ensure resource cleanup when program exits"""
    logger.info(f"Received signal {signum}, shutting down server...")
    print(f"\nReceived signal {signum}, shutting down server...")
    cleanup_model()
    sys.exit(0)


def main():
    global standby_mode
    
    logger.info("UI-Ins Server starting up")
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Mode: {UI_INS_MODE.upper()}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="UI-Ins Server - OpenAI-compatible UI-INS API server (online or local mode)")
    parser.add_argument(
        "--mode", 
        choices=["online", "local"], 
        default=UI_INS_MODE,
        help=f"Execution mode: online (proxy to remote API) or local (run inference locally) (current: {UI_INS_MODE})"
    )
    parser.add_argument(
        "--standby-mode", 
        choices=["cold", "hot"], 
        default="cold",
        help="Standby mode (local mode only): cold (unload model after each request) or hot (preload on startup)"
    )
    parser.add_argument(
        "--host", 
        default=UI_INS_SERVER_HOST,
        help=f"Server host address (default: {UI_INS_SERVER_HOST})"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=UI_INS_SERVER_PORT,
        help=f"Server port (default: {UI_INS_SERVER_PORT})"
    )
    parser.add_argument(
        "--model-path", 
        default=UI_INS_MODEL_PATH,
        help=f"UI-Ins model path (local mode only, default: {UI_INS_MODEL_PATH})"
    )
    parser.add_argument(
        "--remote-api", 
        default=UI_INS_REMOTE_API_URL,
        help=f"Remote UI-INS API URL (online mode only, default: {UI_INS_REMOTE_API_URL})"
    )
    
    args = parser.parse_args()
    logger.debug(f"Command line arguments: mode={args.mode}, standby_mode={args.standby_mode}, host={args.host}, port={args.port}")
    standby_mode = args.standby_mode
    
    # Register cleanup function for program exit
    atexit.register(cleanup_model)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Display mode information
    print("=" * 60)
    print("🤖 UI-Ins Server")
    print("OpenAI-compatible API for UI element localization")
    print("=" * 60)
    print(f"\n📋 Execution Mode: {args.mode.upper()}")
    
    if args.mode == "online":
        logger.info(f"Online mode: Proxying to remote API {args.remote_api}")
        print(f"   Type: Online (Proxy Mode)")
        print(f"   Remote API: {args.remote_api}")
        print(f"   Status: Ready to proxy requests")
    else:  # local mode
        logger.info("Local mode: Running inference locally")
        print(f"   Type: Local (Inference Mode)")
        print(f"   Model Path: {args.model_path}")
        print(f"   Standby Mode: {standby_mode}")
        
        # Decide whether to preload model based on standby mode
        if standby_mode == "hot":
            logger.info("Hot standby mode: Preloading model")
            print(f"   Status: Preloading model...")
            try:
                load_ui_ins_model(args.model_path)
                logger.info("Model preloaded successfully")
                print(f"   Status: ✅ Model loaded and ready")
            except Exception as e:
                logger.error(f"Failed to preload model: {e}", exc_info=True)
                print(f"   Status: ❌ Failed to preload model: {e}")
                print("   Server will start in cold standby mode")
                standby_mode = "cold"
        else:
            logger.info("Cold standby mode: Model will be loaded on first request")
            print(f"   Status: Model will be loaded on first request")
        
        logger.info(f"Standby mode: {standby_mode}")
    
    print("=" * 60)
    
    # Start server
    logger.info(f"Starting UI-Ins Server on http://{args.host}:{args.port}")
    print(f"\n🚀 UI-Ins Server is running on http://{args.host}:{args.port}")
    print("Available endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"\nLogging to: {log_file}")
    print("=" * 60 + "\n")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
