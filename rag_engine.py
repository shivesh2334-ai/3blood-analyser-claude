"""
CBC RAG Engine — Retrieval-Augmented Generation for Clinical CBC Analysis
Embeds UpToDate knowledge chunks using Google Gemini Embeddings,
performs cosine similarity retrieval, and augments Gemini generation.

Updated: gemini-2.0-flash (replaces deprecated gemini-1.5-flash)
"""

import json
import math
import re
import os
import time
from typing import Optional


# ─────────────────────────────────────────────────────────
# VECTOR MATH  (pure Python — no numpy required)
# ─────────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────
# KNOWLEDGE BASE LOADER
# ─────────────────────────────────────────────────────────

def load_knowledge_base(kb_path: str) -> list:
    """Load chunks from JSON knowledge base file."""
    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


# ─────────────────────────────────────────────────────────
# GEMINI EMBEDDING CLIENT
# ─────────────────────────────────────────────────────────

class GeminiEmbeddingClient:
    """Wraps Gemini Embedding API — text-embedding-004."""

    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = model

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
        """Embed a single text string; returns list of floats."""
        result = self.genai.embed_content(
            model=self.model,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]

    def embed_query(self, text: str) -> list:
        """Embed a query string (RETRIEVAL_QUERY task type)."""
        return self.embed(text, task_type="RETRIEVAL_QUERY")

    def embed_batch(self, texts: list, task_type: str = "RETRIEVAL_DOCUMENT",
                    delay: float = 0.1) -> list:
        """Embed a list of texts with rate-limit delay."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text, task_type=task_type))
            time.sleep(delay)
        return embeddings


# ─────────────────────────────────────────────────────────
# VECTOR STORE  (in-memory)
# ─────────────────────────────────────────────────────────

class InMemoryVectorStore:
    """Lightweight in-memory vector store using cosine similarity."""

    def __init__(self):
        self.documents  = []   # list of chunk dicts
        self.embeddings = []   # parallel list of embedding vectors

    def add(self, chunk: dict, embedding: list):
        self.documents.append(chunk)
        self.embeddings.append(embedding)

    def search(self, query_embedding: list, top_k: int = 5,
               section_filter: Optional[str] = None) -> list:
        """Return top_k chunks by cosine similarity, with optional section filter."""
        scored = []
        for idx, emb in enumerate(self.embeddings):
            doc = self.documents[idx]
            if section_filter and doc.get("section") != section_filter:
                continue
            score = cosine_similarity(query_embedding, emb)
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            chunk = self.documents[idx].copy()
            chunk["_score"] = round(score, 4)
            results.append(chunk)
        return results

    def __len__(self):
        return len(self.documents)


# ─────────────────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────────────────

class CBCRagEngine:
    """
    Main RAG engine for CBC clinical analysis.

    Workflow:
      1. Load knowledge base chunks (JSON)
      2. Build embeddings once per session (cached in Streamlit session_state)
      3. For each clinical query: embed query → retrieve chunks → generate answer
    """

    def __init__(self, api_key: str, kb_path: str,
                 embed_model: str = "models/text-embedding-004",
                 gen_model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai     = genai
        self.gen_model = gen_model
        self.embedder  = GeminiEmbeddingClient(api_key=api_key, model=embed_model)
        self.store     = InMemoryVectorStore()
        self.chunks    = load_knowledge_base(kb_path)
        self._ready    = False

    # ── INDEX BUILDING ────────────────────────────────────

    def build_index(self, progress_callback=None) -> int:
        """
        Embed all chunks and load into the vector store.
        progress_callback(i, total) is called after each chunk.
        Returns number of chunks indexed.
        """
        total = len(self.chunks)
        for i, chunk in enumerate(self.chunks):
            # Rich embedding: section + title + keywords + body text
            text_to_embed = (
                f"Section: {chunk.get('section', '')}\n"
                f"Title: {chunk.get('title', '')}\n"
                f"Keywords: {', '.join(chunk.get('keywords', []))}\n"
                f"{chunk['text']}"
            )
            embedding = self.embedder.embed(text_to_embed, task_type="RETRIEVAL_DOCUMENT")
            self.store.add(chunk, embedding)
            if progress_callback:
                progress_callback(i + 1, total)
            time.sleep(0.05)   # gentle rate-limit

        self._ready = True
        return total

    def is_ready(self) -> bool:
        return self._ready

    # ── RETRIEVAL ─────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4,
                 section_filter: Optional[str] = None) -> list:
        """Retrieve top_k most relevant knowledge chunks for a query."""
        if not self._ready:
            raise RuntimeError("Index not built. Call build_index() first.")
        q_emb = self.embedder.embed_query(query)
        return self.store.search(q_emb, top_k=top_k, section_filter=section_filter)

    def format_context(self, chunks: list) -> str:
        """Format retrieved chunks as a context block for the generation prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk.get('section', '')} — {chunk.get('title', '')} "
                f"(relevance: {chunk.get('_score', 0):.3f})]\n{chunk['text']}"
            )
        return "\n\n".join(parts)

    # ── GENERATE WITH RAG ─────────────────────────────────

    def generate_with_rag(self, query: str, top_k: int = 4,
                          additional_context: str = "",
                          temperature: float = 0.2) -> dict:
        """
        Full RAG pipeline: retrieve → augment prompt → generate answer.
        Returns dict: {answer, sources, retrieved_chunks, query}
        """
        # 1. Retrieve
        retrieved   = self.retrieve(query, top_k=top_k)
        context_str = self.format_context(retrieved)

        # 2. Build augmented prompt
        extra = (
            f"ADDITIONAL PATIENT CONTEXT:\n{additional_context}\n"
            if additional_context else ""
        )
        prompt = f"""You are an expert clinical hematologist with deep knowledge of CBC \
interpretation per UpToDate guidelines.

Use ONLY the following retrieved knowledge passages to answer the clinical question. \
Cite each source as [Source N] inline. If the passages do not contain enough information, \
clearly state this.

────────────────────────────────────────
RETRIEVED KNOWLEDGE PASSAGES:
{context_str}
────────────────────────────────────────

{extra}
CLINICAL QUESTION:
{query}

────────────────────────────────────────
INSTRUCTIONS:
- Answer with clinical precision grounded only in the provided passages
- Use [Source N] citations inline for every clinical claim
- Structure your answer: Clinical Finding | Interpretation | Differential Diagnosis | Recommended Next Steps
- Be specific about cutoff values, mechanisms, and test recommendations
- End with a "Key References Used" list
"""

        # 3. Generate using gemini-2.0-flash
        model = self.genai.GenerativeModel(
            self.gen_model,
            generation_config={"temperature": temperature, "max_output_tokens": 1500}
        )
        response = model.generate_content(prompt)
        answer   = response.text

        # 4. Build source attribution list
        sources = [
            {
                "index":   i + 1,
                "title":   c.get("title", ""),
                "section": c.get("section", ""),
                "score":   c.get("_score", 0),
                "preview": c["text"][:120] + "..."
            }
            for i, c in enumerate(retrieved)
        ]

        return {
            "answer":           answer,
            "sources":          sources,
            "retrieved_chunks": retrieved,
            "query":            query,
        }

    # ── TARGETED ANALYSIS METHODS ─────────────────────────

    def analyze_anemia(self, cbc_values: dict, sex: str) -> Optional[dict]:
        hgb   = cbc_values.get("hgb")
        mcv   = cbc_values.get("mcv")
        rdw   = cbc_values.get("rdw")
        retic = cbc_values.get("retic")
        hgb_lo = 13.5 if sex == "M" else 12.0

        if hgb is None or hgb >= hgb_lo:
            return None

        query = (
            f"Patient: {sex}, Hgb={hgb} g/dL, MCV={mcv} fL, RDW={rdw}%, "
            f"Reticulocytes={retic}%.\n"
            "Classify the type of anemia (microcytic/normocytic/macrocytic), identify the "
            "most likely causes, explain the pathophysiology, and recommend specific next "
            "investigations. What does the RDW tell us? What is the reticulocyte production "
            "index indicating?"
        )
        return self.generate_with_rag(
            query=query,
            top_k=5,
            additional_context=f"CBC values: {json.dumps({k:v for k,v in cbc_values.items() if v})}",
        )

    def analyze_neutrophil_abnormality(self, cbc_values: dict) -> Optional[dict]:
        wbc       = cbc_values.get("wbc")
        neut_abs  = cbc_values.get("neut_abs")
        neut_pct  = cbc_values.get("neut_pct")
        bands     = cbc_values.get("bands")

        anc = neut_abs
        if anc is None and wbc and neut_pct:
            anc = wbc * neut_pct / 100

        if anc is None:
            return None

        if anc > 7.7:
            condition = "neutrophilia"
            query = (
                f"WBC={wbc} ×10⁹/L, ANC={anc:.2f} ×10⁹/L, Bands={bands}%.\n"
                "Evaluate this neutrophilia. Classify severity, enumerate the most likely "
                "causes in order, differentiate reactive from neoplastic causes, indicate "
                "red flags for CML or other MPNs, and provide specific workup steps."
            )
        elif anc < 1.8:
            condition = "neutropenia"
            query = (
                f"WBC={wbc} ×10⁹/L, ANC={anc:.2f} ×10⁹/L.\n"
                "Evaluate this neutropenia. Classify severity and infection risk, list common "
                "causes by category (drugs, autoimmune, congenital, infectious, marrow failure), "
                "and provide a stepwise evaluation approach."
            )
        else:
            return None

        return self.generate_with_rag(
            query=query,
            top_k=4,
            additional_context=f"Condition: {condition}. CBC: {json.dumps({k:v for k,v in cbc_values.items() if v})}",
        )

    def analyze_platelet_abnormality(self, cbc_values: dict) -> Optional[dict]:
        plt = cbc_values.get("plt")
        mpv = cbc_values.get("mpv")

        if plt is None:
            return None

        if plt < 150:
            query = (
                f"Platelet count={plt} ×10⁹/L, MPV={mpv} fL.\n"
                "Evaluate this thrombocytopenia. Address pseudothrombocytopenia first. "
                "Classify severity. Use MPV to guide differential diagnosis. "
                "List the most common causes and their distinguishing features. "
                "What urgent steps are needed if platelets are critically low?"
            )
        elif plt > 400:
            query = (
                f"Platelet count={plt} ×10⁹/L.\n"
                "Evaluate this thrombocytosis. Distinguish reactive from clonal causes. "
                "At what threshold should primary thrombocytosis be suspected? "
                "What mutations should be tested and when?"
            )
        else:
            return None

        return self.generate_with_rag(
            query=query,
            top_k=4,
            additional_context=f"CBC: {json.dumps({k:v for k,v in cbc_values.items() if v})}",
        )

    def analyze_immunodeficiency_risk(self, cbc_values: dict, sex: str, age: int) -> dict:
        lymph_abs = cbc_values.get("lymph_abs")
        lymph_pct = cbc_values.get("lymph_pct")
        wbc       = cbc_values.get("wbc")
        neut_abs  = cbc_values.get("neut_abs")
        plt       = cbc_values.get("plt")
        mpv       = cbc_values.get("mpv")

        alc = lymph_abs
        if alc is None and wbc and lymph_pct:
            alc = wbc * lymph_pct / 100

        query = (
            f"Patient: {sex}, age {age} years. ALC={alc} ×10⁹/L, ANC={neut_abs} ×10⁹/L, "
            f"Platelets={plt} ×10⁹/L, MPV={mpv} fL.\n"
            "Screen this CBC for primary immunodeficiency disorders (PID). "
            "What patterns in the CBC suggest specific PIDs? "
            "For this patient, what are the red flags and which conditions should be ruled out first? "
            "Describe the stepwise evaluation including flow cytometry and immunoglobulin testing."
        )
        return self.generate_with_rag(
            query=query,
            top_k=4,
            additional_context=f"Patient sex: {sex}, age: {age}. CBC: {json.dumps({k:v for k,v in cbc_values.items() if v})}",
        )

    def full_rag_analysis(self, cbc_values: dict, sex: str, age: int) -> dict:
        """Comprehensive RAG-based clinical narrative covering all abnormal findings."""
        entered = {k: v for k, v in cbc_values.items() if v is not None and v > 0}

        hgb_lo = 13.5 if sex == "M" else 12.0
        hgb_hi = 17.5 if sex == "M" else 15.5

        abnormals = []
        hgb = cbc_values.get("hgb")
        if hgb:
            if hgb < hgb_lo: abnormals.append(f"Anemia (Hgb {hgb} g/dL)")
            if hgb > hgb_hi: abnormals.append(f"Erythrocytosis (Hgb {hgb} g/dL)")

        wbc = cbc_values.get("wbc")
        if wbc:
            if wbc > 11.0: abnormals.append(f"Leukocytosis (WBC {wbc} ×10⁹/L)")
            if wbc < 4.5:  abnormals.append(f"Leukopenia (WBC {wbc} ×10⁹/L)")

        neut = cbc_values.get("neut_abs")
        if neut:
            if neut > 7.7: abnormals.append(f"Neutrophilia (ANC {neut})")
            if neut < 1.8: abnormals.append(f"Neutropenia (ANC {neut})")

        plt = cbc_values.get("plt")
        if plt:
            if plt < 150: abnormals.append(f"Thrombocytopenia (PLT {plt})")
            if plt > 400: abnormals.append(f"Thrombocytosis (PLT {plt})")

        lymph = cbc_values.get("lymph_abs")
        if lymph:
            if lymph < 1.0: abnormals.append(f"Lymphopenia (ALC {lymph})")
            if lymph > 4.8: abnormals.append(f"Lymphocytosis (ALC {lymph})")

        mcv = cbc_values.get("mcv")
        if mcv:
            if mcv < 80:  abnormals.append(f"Microcytosis (MCV {mcv} fL)")
            if mcv > 100: abnormals.append(f"Macrocytosis (MCV {mcv} fL)")

        sex_word = "male" if sex == "M" else "female"
        abn_str  = "; ".join(abnormals) if abnormals else "None identified"

        query = (
            f"Complete CBC analysis for a {age}-year-old {sex_word} patient.\n"
            f"Identified abnormalities: {abn_str}.\n"
            f"CBC values: {json.dumps(entered, indent=2)}.\n\n"
            "Provide a comprehensive clinical analysis:\n"
            "1. PRIORITIZED FINDINGS: Rank abnormalities by clinical urgency\n"
            "2. UNIFIED DIFFERENTIAL: What single diagnosis or combination best explains all findings?\n"
            "3. PATHOPHYSIOLOGY: Link the CBC pattern to underlying mechanisms\n"
            "4. CRITICAL ALERTS: Any values requiring immediate action?\n"
            "5. SEQUENTIAL INVESTIGATION PLAN: Step-by-step with rationale\n"
            "6. COLLECTION QUALITY: Any CBC internal inconsistencies suggesting pre-analytical error?"
        )
        return self.generate_with_rag(
            query=query,
            top_k=6,
            additional_context=f"Patient: {sex}, age {age}. Abnormalities: {abnormals}",
        )


# ─────────────────────────────────────────────────────────
# KEYWORD FALLBACK RETRIEVAL  (no API required)
# ─────────────────────────────────────────────────────────

class KeywordRetriever:
    """
    TF-IDF-style keyword overlap retriever.
    Used as a fallback when no Gemini API key is available.
    """

    def __init__(self, chunks: list):
        self.chunks = chunks

    def _score(self, chunk: dict, query_tokens: set) -> float:
        chunk_text = (
            chunk["text"].lower() + " " +
            chunk.get("title", "").lower() + " " +
            " ".join(chunk.get("keywords", [])).lower()
        )
        chunk_tokens = set(re.findall(r"\b\w+\b", chunk_text))
        overlap = query_tokens & chunk_tokens
        if not chunk_tokens:
            return 0.0
        return len(overlap) / math.sqrt(len(chunk_tokens))

    def search(self, query: str, top_k: int = 4) -> list:
        query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        scored = [(self._score(c, query_tokens), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            c = chunk.copy()
            c["_score"]  = round(score, 4)
            c["_method"] = "keyword"
            results.append(c)
        return results


# ─────────────────────────────────────────────────────────
# FACTORY HELPERS
# ─────────────────────────────────────────────────────────

def create_rag_engine(api_key: str, kb_path: str) -> CBCRagEngine:
    """Create and return a configured CBCRagEngine."""
    return CBCRagEngine(api_key=api_key, kb_path=kb_path)


def create_keyword_retriever(kb_path: str) -> KeywordRetriever:
    """Create and return a keyword-based fallback retriever."""
    chunks = load_knowledge_base(kb_path)
    return KeywordRetriever(chunks)
