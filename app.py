# Import libraries
from Run_Query import rag_llm_pipeline
import streamlit as st

# Streamlit
st.set_page_config(
    page_title="Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("Chatbot Portal da Transparência")
    page = st.radio("Navegação", ["Chat", "Sobre"])

if page == "Chat":
    st.title("Chat")

    # Welcome message
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("Bem vindo ao Chatbot sobre o portal da transparência!👋")
    
    # User input
    user_input = st.chat_input("Digite sua pergunta...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Execute RAG + LLM pipeline
        try:
            result = rag_llm_pipeline(user_input)
            st.session_state.messages.append({
                    "role": "assistant",
                    "content": str(result)})
        # Show error            
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Erro: {e}"})

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Clean chat history button
    if st.button("Limpar Chat"):
        st.session_state.messages = []

# "About"
elif page == "Sobre":
    st.title("Sobre")
    st.write(
        "Este é um chatbot em português brasileiro baseado no faq do portal da transparência (https://portaldatransparencia.gov.br/perguntas-frequentes).")