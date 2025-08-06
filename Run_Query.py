# Import libraries
import os
import re
import nltk
from nltk.corpus import stopwords
import chromadb
from Create_Chromadb import HFEmbeddingFunction
from langchain_ollama import OllamaLLM

# Download stopwords
nltk.download('stopwords', quiet=True)
_stopwords_pt = set(stopwords.words('portuguese'))

# Connect to ChromaDB
db_path = os.path.join(os.getcwd(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="qa_collection")

# Embedding
embed_fn = HFEmbeddingFunction()

# Extract keywords
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)
    kw = [w for w in words if w not in _stopwords_pt and len(w) > 3]
    return list(dict.fromkeys(kw))

# LLM Query
def query_ollama(prompt):
    llm = OllamaLLM(model="llama3:latest", temperature=0.0, base_url="http://host.docker.internal:11434") # Connect outside of the container
    return llm.invoke(prompt)

# RAG + LLM pipeline
def rag_llm_pipeline(query_text, n_results=5):
    # Embedding
    query_emb = embed_fn(query_text)

    # Search top-N results
    results = collection.query(
        query_embeddings=query_emb,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Re-ranking with the presence of keywords
    qkws = set(extract_keywords(query_text))
    scored = []
    for doc, meta, dist in zip(docs, metas, dists):
        sim = 1.0 - dist
        kws = meta.get("palavras_chave", "").split(
            ",") if meta.get("palavras_chave") else []
        bonus = len(qkws.intersection(kws)) * 0.1
        scored.append((doc, meta, dist, sim + bonus, kws))

    scored.sort(key=lambda x: x[3], reverse=True)
    doc, meta, dist, score, kws = scored[0]
    sim = 1.0 - dist

    # Prompt LLM
    instruction = (
        "Você é um assistente que SOMENTE pode usar as informações do contexto abaixo. "
        "Responda estritamente com base nesse contexto e inclua todos os detalhes fornecidos, sem omitir nenhuma parte, especialmente todos os itens numerados, quebras de linha e emojis. "
    )

    prompt = (
        f"{instruction}"
        f"Contexto: [Pergunta indexada: '{doc}'. Resposta: '\'\'{meta.get('resposta', '')}\'\'\']\n\nPergunta do usuário: '{query_text} Responda somente baseado no conteúdo do contexto'\nResposta:"
    )
    print("######## Augmented Prompt ########")
    print(prompt)
    return query_ollama(prompt)


# Single test
if __name__ == "__main__":
    sample = "Por que ao abir um arquivo CSV baixado do portal da transparência, o meu aplicativo de planilha eletrônica alerta que ultrapassou o limite de linhas? Que dados serão perdidos?"
    print("\n💬 Resposta do LLM:\n", rag_llm_pipeline(sample))