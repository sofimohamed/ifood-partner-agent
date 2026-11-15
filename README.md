# Partner Ops AI — Assistente Inteligente de Operações
 **Aplicação Online (Live Demo):** [Acessar Partner Ops AI](https://ifood-partner-agent-9nbuysr7gyzgqho95vti7l.streamlit.app)
 
Agente de IA conversacional projetado para apoiar restaurantes parceiros em dúvidas operacionais, financeiras e logísticas, combinando **RAG (Retrieval-Augmented Generation)**, **Guardrails anti-alucinação**, **Resiliência multi-modelo** e **Observabilidade LLMOps**.

---

## Arquitetura e Decisões Técnicas

```text
[Usuário / Interface Streamlit]
        │
        ▼
[PartnerOpsEngine (engine.py)]
        ├──► 1. Recuperação Semântica (TF-IDF + Similaridade de Cosseno)
        │         └── Consulta base de conhecimento (politicas_ops.txt)
        ├──► 2. Injeção de Contexto + Guardrails Estritos
        │         └── Bloqueia respostas fora da base de políticas
        ├──► 3. Orquestração Resiliente (Retry + Multi-Model Fallback)
        │         └── gemini-3.8-flash ──► gemini-3.8-flash-lite ──► gemini-3.5-flash
        └──► 4. Telemetria de Inferência (LLMOps)
                  └── Latência (ms), contagem de tokens e scores de recuperação
```
 
