# LangChain Multiple Q&A Chatbot with RAG

This repository contains a Retrieval-Augmented Generation (RAG) chatbot application built with Langchain, using the Hugging Face LLAMA-3 model and various tools to retrieve and process information from different sources including Wikipedia, web pages, and PDF documents.
## Tech Stack
- **Python**: Main programming language.
- **Langchain**: Library to manage prompts, language models, and tool integration.
- **FAISS**: Facebook AI Similarity Search for efficient similarity search and clustering of dense vectors.
- **Hugging Face**: Platform providing various NLP models and embeddings.
- **dotenv**: Library for managing environment variables.
## Features
- **Retrieval-Augmented Generation**: Combines retrieval of relevant documents with generation of natural language responses.
- **Integration with Langchain**: Manages prompts, language models, and tool integration.
- **Hugging Face Embeddings**: Utilizes Hugging Face embedding models to convert text into vectors.
- **Document Loaders and Text Splitters**: Loads and processes text from PDF files and web pages.
- **Custom Retrievers**: Retrieves relevant information using FAISS for embedded vectors.
- **Agent Execution**: Executes agents to respond to user queries using integrated tools.
## Setup Instructions
### Prerequisites
Ensure you have the following installed:
- Python 3.8 or above
### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Aman3786/LangChain-Multiple-QA-chatbot-with-RAG.git
   cd LangChain-Multiple-QA-chatbot-with-RAG
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the root directory with the following content:
     ```env
     HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token"
     LANGCHAIN_API_KEY="your_langchain_api_key"
     ```
### Running the Code
To run the script and start the chatbot, follow these steps:
1. **Execute the script**:
   ```sh
   python Q&A_RAG.py
   ```
2. **Interact with the chatbot**:
   - The script will prompt you to enter a search query.
   - The chatbot will process the query using the LLAMA-3 model and integrated tools to fetch and display relevant information.

### Acknowledgements
- [Langchain](https://python.langchain.com/v0.2/docs/introduction/) for the core library.
- [Hugging Face](https://huggingface.co/) for models and embedding tools.
