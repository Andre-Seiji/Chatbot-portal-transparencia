# Chatbot-portal-transparencia

https://github.com/user-attachments/assets/a9d32cc9-970e-4a8e-857c-7776a99bbcc5

A portuguese chatbot for answering questions about "Portal da transparência" (https://portaldatransparencia.gov.br/perguntas-frequentes). It uses Streamlit, and RAG + LLM pipeline with Docker and runs Ollama locally.

The embedding is made with GTE model (https://huggingface.co/Alibaba-NLP/gte-multilingual-base) and it is stored in the Chromadb.

Adapting this chatbot to other contexts is simple: update the FAQ file and the stopwords, and modify the LLM's prompt.

## Installation

1. Download Ollama (https://ollama.com/download/);
2. Download Docker (https://www.docker.com/products/docker-desktop/) and Docker Compose (https://docs.docker.com/compose/install/);
3. Run "docker-compose up --build".
