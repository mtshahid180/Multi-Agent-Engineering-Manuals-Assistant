"""Model configuration: ek hosted LLM (Groq) + Google embeddings."""
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value or value.startswith("apni_"):
        raise RuntimeError(f"{name} missing. .env file mein key daalo.")
    return value


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Groq chat model. temperature=0 -> factual, deterministic jawab."""
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        api_key=_require("GROQ_API_KEY"),
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Google embeddings: text -> numbers (vector).

    embed_documents() RETRIEVAL_DOCUMENT aur embed_query() RETRIEVAL_QUERY
    task type khud use karta hai, jo search ke liye behtar hai.
    """
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=_require("GOOGLE_API_KEY"),
    )
def structured_chain(llm, schema, system: str, human: str):
    """prompt | LLM jo seedha Pydantic object (schema) wapas deta hai.

    Agents isi se "free text" ke bajaye validated, structured jawab lete hain.
    """
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm.with_structured_output(schema)