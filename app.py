import streamlit as st
from engine import PartnerOpsEngine

st.set_page_config(page_title="Partner Ops AI Agent", page_icon="🛵", layout="wide")

@st.cache_resource
def load_engine():
    return PartnerOpsEngine()

st.title("🛵 Partner Ops AI — Assistente de Operações")
st.caption("Arquitetura RAG (TF-IDF + Gemini 3.8 Flash) com Guardrails, Fallback e Telemetria LLMOps")

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Falha ao inicializar engine: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_telemetry" not in st.session_state:
    st.session_state.last_telemetry = None

with st.sidebar:
    st.header("⚙️ Observabilidade (LLMOps)")
    if st.session_state.last_telemetry:
        tel = st.session_state.last_telemetry
        col1, col2 = st.columns(2)
        col1.metric("Latência", f"{tel['latency_ms']} ms")
        col2.metric("Tokens Totais", tel["prompt_tokens"] + tel["output_tokens"])
        
        st.markdown(f"**Scores RAG (Cosseno):** `{tel['similarity_scores']}`")
        with st.expander("Contexto Injetado no Prompt"):
            st.text(tel["context_used"])
    else:
        st.info("Envie uma mensagem ou clique em um cenário abaixo para visualizar métricas de inferência.")

    st.divider()
    if st.button("Limpar Sessão", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_telemetry = None
        st.rerun()

# Botões interativos de teste rápido
st.markdown("**Teste rápido (cenários operacionais e guardrail):**")
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

quick_prompt = None
if col_btn1.button("⏱️ Atraso > 45 min", use_container_width=True):
    quick_prompt = "Meu pedido atrasou 50 minutos, vou receber o repasse?"
if col_btn2.button("📦 Vazamento de líquido", use_container_width=True):
    quick_prompt = "O cliente reclamou de vazamento na bebida com foto, o valor será descontado?"
if col_btn3.button("💰 Data do repasse", use_container_width=True):
    quick_prompt = "Quando caem os repasses da semana e qual a taxa de antecipação?"
if col_btn4.button("🚫 Fora de escopo", use_container_width=True):
    quick_prompt = "Qual o valor da taxa de entrega cobrada do cliente?"

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Digite sua dúvida operacional aqui...")
prompt = quick_prompt or user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando políticas e gerando resposta..."):
            result = engine.generate_response(prompt, st.session_state.messages[:-1])
            st.markdown(result["answer"])
            st.session_state.last_telemetry = result

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.rerun()