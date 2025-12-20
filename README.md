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
- Dependencies: `requests`, `Pillow`, `llama-index`, `beautifulsoup4`

### Install Dependencies
```bash
pip install -r requirements_ops_cli.txt
```

### Configuration Files
- Translation files are located in the `i18n` directory
- RAG Configuration:
  - `RAG_API_BASE`: Local API address (default: `http://localhost:11434/v1`)
  - `RAG_EMBED_MODEL`: Embedding model (default: `qwen3-embedding:0.6b`)
  - `RAG_INDEX_DIR`: Index storage directory (default: `./index`)
  - `RAG_DOCS_DIR`: Document directory (default: `./docs`)

## Usage Guide

### Starting the Application

```bash
python ops_cli.py
```

### Initial Configuration

When you first run the application, you will be prompted to configure:

1. **API URL**: Default is `http://localhost:11434/v1/chat/completions`
2. **VLM Model**: Default is "qwen3-vl:32b"
3. **UI-Ins API URL**: Default is `http://localhost:2345/v1/chat/completions`
4. **UI-Ins Model**: Default is "ui-ins-7b"

*Note: The recommended online VLM model is "qwen3-vl-32b-thinking" on ModelScope.
*Note: The UI-Ins model is used for element localization in image-based conversations. If you don't have a local UI-Ins server, you can use the model "gui-plus" on ModelScope.

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