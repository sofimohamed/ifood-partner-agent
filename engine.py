import os
import time
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

class PartnerOpsEngine:
    def __init__(self, knowledge_path: str = "politicas_ops.txt"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não encontrada no ambiente.")
        self.client = genai.Client(api_key=api_key)
        self.models = ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.5-flash"]
        self.chunks = self._load_and_chunk(knowledge_path)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def _load_and_chunk(self, path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

    def retrieve_context(self, query: str, top_k: int = 2) -> tuple[str, list[float]]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        retrieved = []
        scores = []
        for idx in top_indices:
            if similarities[idx] > 0.05:
                retrieved.append(self.chunks[idx])
                scores.append(round(float(similarities[idx]), 3))
        
        context_text = "\n\n".join(retrieved) if retrieved else "Nenhuma política específica encontrada."
        return context_text, scores

    def generate_response(self, user_query: str, chat_history: list[dict]) -> dict:
        start_time = time.perf_counter()
        context, scores = self.retrieve_context(user_query)

        system_instruction = f"""
        Você é um Engenheiro de Operações e Suporte a Restaurantes Parceiros (estilo iFood Partner Ops).
        Sua missão é responder dúvidas operacionais, financeiras e logísticas com precisão.

        REGRAS ESTRITAS (GUARDRAILS):
        1. Responda EXCLUSIVAMENTE com base no CONTEXTO DE POLÍTICAS fornecido abaixo.
        2. Se a informação não estiver no contexto, diga claramente: "Não encontrei essa regra nas políticas vigentes. Recomendo abrir um chamado técnico." Não invente prazos ou valores.
        3. Estruture a resposta em:
           - **Categoria:** (Logística / Financeiro / Operação / Fora de Escopo)
           - **Orientação:** Resposta direta e objetiva.
           - **Próximo Passo:** Ação prática recomendada ao parceiro.

        CONTEXTO DE POLÍTICAS RECUPERADO:
        {context}
        """

        formatted_history = [
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])]
            )
            for msg in chat_history[-6:]
        ]
        formatted_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
        )

        last_error = ""
        for model_name in self.models:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=formatted_history,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.2,
                        )
                    )
                    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
                    usage = response.usage_metadata

                    return {
                        "answer": f"*(Modelo: `{model_name}`)*\n\n{response.text}",
                        "context_used": context,
                        "similarity_scores": scores,
                        "latency_ms": latency_ms,
                        "prompt_tokens": usage.prompt_token_count if usage else 0,
                        "output_tokens": usage.candidates_token_count if usage else 0,
                    }
                except Exception as e:
                    last_error = str(e)
                    if "503" in last_error and attempt == 0:
                        time.sleep(2)
                        continue
                    break

        return {
            "answer": f"Erro na comunicação com os LLMs disponíveis: {last_error}",
            "context_used": context,
            "similarity_scores": scores,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 1),
            "prompt_tokens": 0,
            "output_tokens": 0,
        }