# Import libraries
import os
import shutil
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.api.types import Documents, Embeddings

# Embedding model from Huggingface
class HFEmbeddingFunction:
    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-base", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model     = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def __call__(self, input: Documents) -> Embeddings:
        texts = input if isinstance(input, list) else [input]
        enc   = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        embs = out.last_hidden_state.mean(dim=1)
        embs = F.normalize(embs, p=2, dim=1)
        return embs.cpu().tolist()

def create_chromadb(file):
    db_path = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(db_path, exist_ok=True)

    # Delete former database
    if os.path.exists(db_path):
        for entry in os.listdir(db_path):
            full = os.path.join(db_path, entry)
            if os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.remove(full)

    # Inicializa client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Load FAQ
    with open(file, "r", encoding="utf-8") as f:
        faq_entries = json.load(f)["base_conhecimento"]

    documents = []
    metadatas = []
    ids = []

    for id, entry in enumerate(faq_entries):
        pergunta = entry.get("pergunta")
        resposta = entry["resposta"]
        kws_list = entry.get("palavras_chave", [])
        kws_str  = ",".join(kws_list)

        documents.append(pergunta[0])
        metadatas.append({
                "id": id,
                "resposta": resposta,
                "palavras_chave": kws_str
        })
        ids.append(str(id))

    # Create vectorial collection
    embedding_fn = HFEmbeddingFunction()
    collection   = chroma_client.get_or_create_collection(
        name="qa_collection",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_fn
    )

    # Add documents
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("✅ ChromaDB created!")

if __name__ == "__main__":
    create_chromadb("faq_portal_transparencia.json")