# Openterface Ops CLI

## Overview

Openterface Ops CLI is an intelligent dialogue client that integrates Openterface AI Chat Client and UI-Ins model functionality, providing the following core features:

### 1. Multilingual Support
- Supports Chinese and English switching
- Built-in internationalization translation system
- Automatically loads corresponding language translation files

### 2. RAG (Retrieval-Augmented Generation) Functionality
- Build vector indexes from document directories
- Load pre-built indexes
- Retrieve relevant documents based on queries
- Incorporate retrieval results into conversation context

### 3. Image Acquisition and Processing
- Get the latest image from Openterface server
- Support image encoding to base64 format
- Automatically save images to specified directory

### 4. Multi-turn Conversation
- Support multi-turn conversation mode
- Maintain conversation history
- Context-aware intelligent responses

### 5. API Integration
- Compatible with OpenAI format APIs
- Support local AI model deployment
- Support multi-modal conversations with image input

## Installation and Configuration

### Environment Requirements
- Python 3.12
- Dependencies: `requests`, `Pillow`, `llama-index`, `beautifulsoup4`, `python-dotenv`

### Install Dependencies
```bash
pip install -r requirements_ops_cli.txt
```

### Environment Configuration

The application uses a `.env` file for configuration. This allows you to customize all settings without modifying the code.

#### Setup .env File

1. Copy the example configuration:
```bash
cp .env.example .env
```

2. Edit the `.env` file with your specific settings:
```bash
nano .env
# or use your preferred editor
```

#### Configuration Sections

The `.env` file is organized into 5 main sections:

**1. Openterface Device Configuration**
```
OPENTERFACE_HOST=localhost      # Device IP or hostname
OPENTERFACE_PORT=12345          # TCP port for device connection
```

**2. Main AI API Configuration**
```
API_URL=http://localhost:11434/v1/chat/completions    # Your LLM API endpoint
MODEL=qwen3-vl:32b                                      # Model name to use
```

**3. UI-Ins Element Localization Model**
```
UI_INS_API_URL=http://localhost:2345/v1/chat/completions  # UI-Ins API endpoint
UI_INS_MODEL=ui-ins-7b                                      # UI element detection model
```

**4. API Authentication**
```
API_KEY=EMPTY                   # Set to your API key if authentication is required
```

**5. RAG Configuration**
```
RAG_API_BASE=http://localhost:11434/v1
RAG_EMBED_MODEL=qwen3-embedding:0.6b
RAG_INDEX_DIR=./index           # Directory to store built indexes
RAG_DOCS_DIR=./docs             # Directory containing source documents
```

#### Configuration Files
- Translation files are located in the `i18n` directory
- Configuration template: `.env.example` (reference for all available options)
- Configuration file: `.env` (your custom settings, not tracked by git)

#### Common Configuration Examples

**Using Ollama locally:**
```
API_URL=http://localhost:11434/v1/chat/completions
MODEL=qwen3-vl:32b
```

**Using vLLM:**
```
API_URL=http://localhost:8000/v1/chat/completions
MODEL=your-model-name
```

**Using LM Studio:**
```
API_URL=http://localhost:1234/v1/chat/completions
MODEL=local-model
```

**Using remote device (not localhost):**
```
OPENTERFACE_HOST=192.168.1.100
OPENTERFACE_PORT=12345
```

**With API authentication:**
```
API_KEY=your-actual-api-key
```

## Usage Guide

### Starting the Application

```bash
python ops_cli.py
```

### Initial Setup

Before running the application for the first time:

1. **Install dependencies:**
   ```bash
   pip install -r requirements_ops_cli.txt
   ```

2. **Configure your environment:**
   ```bash
   cp .env.example .env
   nano .env  # Edit with your actual settings
   ```

3. **Verify your configuration:**
   - Ensure all API endpoints are correctly set
   - Check that device host and port are correct
   - Verify any required API keys are set

4. **Start the application:**
   ```bash
   python ops_cli.py
   ```

### Runtime Configuration

When the application starts, it loads all settings from the `.env` file. You will be prompted to optionally override settings interactively:

1. **API Configuration** - You can accept defaults or enter custom API URL
2. **Model Name** - You can accept the default model or specify a different one
3. **UI-Ins Configuration** - Optional configuration for UI element detection
4. **Connection Test** - The app will verify connectivity to your API endpoints

*Note: Settings in `.env` are used as defaults. Interactive prompts allow you to temporarily override them without editing the file.*

### Interactive Commands

During the conversation, you can use the following commands:

#### Basic Commands
- `/quit` or `/exit` or `/q`: Exit the application
- `/clear` or `/cls`: Clear conversation history
- `/help`: Show help information
- `/info`: Show API information
- `/model`: Switch to a different model

#### Language Commands
- `/lang`: Show current language
- `/lang en`: Switch to English
- `/lang zh`: Switch to Chinese

#### Conversation Mode
- `/multiturn`: Enable multi-turn conversation mode
- `/single`: Switch back to single-turn mode

#### RAG Functionality
- `/load docs`: Load documents and build RAG index

#### Image Processing
- `/image`: Get the latest image from Openterface server and start image-based conversation

### Normal Conversation

1. Enter your question when prompted
2. The AI will process your request and provide a response
3. Continue the conversation by entering new questions

### Multi-turn Conversation

1. Type `/multiturn` to enable multi-turn mode
2. The application will maintain conversation history
3. Responses will be context-aware
4. Type `/single` to exit multi-turn mode

### RAG Usage

1. Type `/load docs` to load documents from the `./docs` directory
2. The application will build a vector index automatically
3. Ask questions related to your documents
4. The AI will use relevant documents to generate more accurate responses

### Image-based Conversation

1. Type `/image` to get the latest image from the server
2. Enter your question about the image
3. The AI will analyze the image and provide a response
4. If the response contains `<click>` tags, the application will automatically process UI element requests

### Example Workflow

```
$ python ops_cli.py

Welcome to Openterface Ops CLI!

Enter API URL (default: http://localhost:11434/v1/chat/completions): 
Enter model name (default: default): 
Connecting to API...
Connection successful!

Enter UI-Ins API URL (default: http://localhost:2345/v1/chat/completions): 
Enter UI-Ins model name (default: default): 
Connecting to UI-Ins API...
Connection successful!

--- Start Chat ---

Your question: Hello

Processing...
Please wait...

AI Response:
----------------------------------------
Hello! How can I assist you today?
----------------------------------------

Your question: /lang zh

Current language: en
Language switched to zh

Your question: 你好

Processing...
Please wait...

AI Response:
----------------------------------------
你好！我能为你做些什么？
----------------------------------------

Your question: /quit

Goodbye!
```
## Notes

1. Ensure the local AI model service is started and running on the specified port
2. Need to build index first when using RAG functionality for the first time
3. Image acquisition functionality requires Openterface server to be running
4. Some features are restricted in multi-turn conversation mode

## License

MIT License

## Contact

For questions or suggestions, please submit an Issue or contact the development team.