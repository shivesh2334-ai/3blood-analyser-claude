"""
CBC RAG Engine — Retrieval-Augmented Generation for Clinical CBC Analysis

Architecture (Claude edition):
  Embeddings : sentence-transformers/all-MiniLM-L6-v2  (local, no API key)
  Generation : Anthropic Claude API  (claude-3-5-haiku-20241022)
  Retrieval  : Cosine similarity (pure Python)
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
# LOCAL SENTENCE-TRANSFORMER EMBEDDER  (no API key needed)
# ─────────────────────────────────────────────────────────

class SentenceTransformerEmbedder:
    """
    Local semantic embedder using sentence-transformers.
    Model: all-MiniLM-L6-v2 (~90 MB, downloaded once, then cached).
    No API key required — runs entirely on device.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str, **kwargs) -> list:
        """Embed a single text string; returns list of floats."""
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list:
        """Embed a search query (same as embed for MiniLM)."""
        return self.embed(text)

    def embed_batch(self, texts: list, **kwargs) -> list:
        """Embed a list of texts; returns list of float lists."""
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


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
        """Return top_k chunks ranked by cosine similarity."""
        scored = []
        for idx, emb in enumerate(self.embeddings):
            doc = self.documents[idx]
            if section_filter and doc.get("section") != section_filter:
                continue
            scored.append((cosine_similarity(query_embedding, emb), idx))

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
      1. Build index (local sentence-transformers embeddings — no API key)
      2. For each clinical query: embed query → cosine search → augment prompt → Claude generates
    """

    DEFAULT_GEN_MODEL = "claude-3-5-haiku-20241022"

    def __init__(self, kb_path: str,
                 api_key: Optional[str] = None,
                 gen_model: str = DEFAULT_GEN_MODEL):
        self.kb_path    = kb_path
        self.api_key    = api_key
        self.gen_model  = gen_model
        self.embedder   = SentenceTransformerEmbedder()
        self.store      = InMemoryVectorStore()
        self.chunks     = load_knowledge_base(kb_path)
        self._ready     = False

    # ── INDEX BUILDING  (local, no API key) ──────────────

    def build_index(self, progress_callback=None) -> int:
        """
        Embed all chunks with local sentence-transformers.
        progress_callback(i, total) called after each chunk.
        Returns number of chunks indexed.
        """
        total = len(self.chunks)

        # Batch embed for efficiency
        texts = []
        for chunk in self.chunks:
            texts.append(
                f"Section: {chunk.get('section', '')}\n"
                f"Title: {chunk.get('title', '')}\n"
                f"Keywords: {', '.join(chunk.get('keywords', []))}\n"
                f"{chunk['text']}"
            )

        embeddings = self.embedder.embed_batch(texts)

        for i, (chunk, emb) in enumerate(zip(self.chunks, embeddings)):
            self.store.add(chunk, emb)
            if progress_callback:
                progress_callback(i + 1, total)

        self._ready = True
        return total

    def is_ready(self) -> bool:
        return self._ready

    # ── RETRIEVAL ─────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4,
                 section_filter: Optional[str] = None) -> list:
        """Retrieve top_k most relevant chunks for a query."""
        if not self._ready:
            raise RuntimeError("Index not built. Call build_index() first.")
        q_emb = self.embedder.embed_query(query)
        return self.store.search(q_emb, top_k=top_k, section_filter=section_filter)

    def format_context(self, chunks: list) -> str:
        """Format retrieved chunks as a context block for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk.get('section', '')} — {chunk.get('title', '')} "
                f"(relevance: {chunk.get('_score', 0):.3f})]\n{chunk['text']}"
            )
        return "\n\n".join(parts)

    # ── GENERATE WITH RAG  (Claude API) ──────────────────

    def generate_with_rag(self, query: str, top_k: int = 4,
                          additional_context: str = "",
                          temperature: float = 0.2) -> dict:
        """
        Full RAG pipeline: retrieve → augment prompt → Claude generates.
        Returns dict: {answer, sources, retrieved_chunks, query}
        """
        if not self.api_key:
            raise ValueError("Anthropic API key required for generation.")

        import anthropic

        # 1. Retrieve relevant chunks
        retrieved   = self.retrieve(query, top_k=top_k)
        context_str = self.format_context(retrieved)

        # 2. Build augmented prompt
        extra = (
            f"ADDITIONAL PATIENT CONTEXT:\n{additional_context}\n\n"
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

{extra}CLINICAL QUESTION:
{query}

────────────────────────────────────────
INSTRUCTIONS:
- Answer with clinical precision grounded only in the provided passages
- Use [Source N] citations inline for every clinical claim
- Structure your answer: Clinical Finding | Interpretation | Differential Diagnosis | Recommended Next Steps
- Be specific about cutoff values, mechanisms, and test recommendations
- End with a "Key References Used" list
"""

        # 3. Generate with Claude
        client  = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.gen_model,
            max_tokens=1500,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = message.content[0].text

        # 4. Build source attribution list
        sources = [
            {
                "index":   i + 1,
                "title":   c.get("title", ""),
                "section": c.get("section", ""),
                "score":   c.get("_score", 0),
                "preview": c["text"][:120] + "…"
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
        hgb    = cbc_values.get("hgb")
        mcv    = cbc_values.get("mcv")
        rdw    = cbc_values.get("rdw")
        retic  = cbc_values.get("retic")
        hgb_lo = 13.5 if sex == "M" else 12.0

        if hgb is None or hgb >= hgb_lo:
            return None

        query = (
            f"Patient: {sex}, Hgb={hgb} g/dL, MCV={mcv} fL, RDW={rdw}%, "
            f"Reticulocytes={retic}%.\n"
            "Classify the anemia (microcytic/normocytic/macrocytic), identify the most likely "
            "causes, explain pathophysiology, and recommend specific next investigations. "
            "What does the RDW indicate? What is the reticulocyte production index?"
        )
        return self.generate_with_rag(
            query=query, top_k=5,
            additional_context=f"CBC: {json.dumps({k:v for k,v in cbc_values.items() if v})}",
        )

    def analyze_neutrophil_abnormality(self, cbc_values: dict) -> Optional[dict]:
        wbc      = cbc_values.get("wbc")
        neut_abs = cbc_values.get("neut_abs")
        neut_pct = cbc_values.get("neut_pct")
        bands    = cbc_values.get("bands")

        anc = neut_abs
        if anc is None and wbc and neut_pct:
            anc = wbc * neut_pct / 100

        if anc is None:
            return None

        if anc > 7.7:
            query = (
                f"WBC={wbc} ×10⁹/L, ANC={anc:.2f} ×10⁹/L, Bands={bands}%.\n"
                "Evaluate neutrophilia: classify severity, enumerate causes in order, "
                "differentiate reactive from neoplastic, flag CML/MPN red flags, "
                "and provide specific workup."
            )
        elif anc < 1.8:
            query = (
                f"WBC={wbc} ×10⁹/L, ANC={anc:.2f} ×10⁹/L.\n"
                "Evaluate neutropenia: classify severity and infection risk, list common causes "
                "by category (drugs, autoimmune, congenital, infectious, marrow failure), "
                "stepwise evaluation approach."
            )
        else:
            return None

        return self.generate_with_rag(
            query=query, top_k=4,
            additional_context=f"CBC: {json.dumps({k:v for k,v in cbc_values.items() if v})}",
        )

    def analyze_platelet_abnormality(self, cbc_values: dict) -> Optional[dict]:
        plt = cbc_values.get("plt")
        mpv = cbc_values.get("mpv")

        if plt is None:
            return None

        if plt < 150:
            query = (
                f"Platelet count={plt} ×10⁹/L, MPV={mpv} fL.\n"
                "Evaluate thrombocytopenia: rule out pseudothrombocytopenia, classify severity, "
                "use MPV to guide DDx, list common causes, urgent steps if critically low."
            )
        elif plt > 400:
            query = (
                f"Platelet count={plt} ×10⁹/L.\n"
                "Evaluate thrombocytosis: distinguish reactive from clonal, threshold for "
                "suspecting primary thrombocytosis, which mutations to test and when."
            )
        else:
            return None

        return self.generate_with_rag(
            query=query, top_k=4,
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
            f"Patient: {sex}, age {age}. ALC={alc} ×10⁹/L, ANC={neut_abs} ×10⁹/L, "
            f"PLT={plt} ×10⁹/L, MPV={mpv} fL.\n"
            "Screen CBC for primary immunodeficiency disorders. What patterns suggest specific PIDs? "
            "Red flags for this patient? Which conditions to rule out first? "
            "Stepwise evaluation including flow cytometry and immunoglobulin testing."
        )
        return self.generate_with_rag(
            query=query, top_k=4,
            additional_context=f"Sex:{sex}, age:{age}. CBC: {json.dumps({k:v for k,v in cbc_values.items() if v})}",
        )

    def full_rag_analysis(self, cbc_values: dict, sex: str, age: int) -> dict:
        """Comprehensive RAG-based clinical narrative for all abnormalities."""
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
            if wbc > 11.0: abnormals.append(f"Leukocytosis (WBC {wbc})")
            if wbc < 4.5:  abnormals.append(f"Leukopenia (WBC {wbc})")

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
            f"Complete CBC analysis for a {age}-year-old {sex_word}.\n"
            f"Abnormalities: {abn_str}.\n"
            f"CBC values: {json.dumps(entered, indent=2)}.\n\n"
            "Provide a comprehensive clinical analysis:\n"
            "1. PRIORITIZED FINDINGS: rank by clinical urgency\n"
            "2. UNIFIED DIFFERENTIAL: best single/combined diagnosis\n"
            "3. PATHOPHYSIOLOGY: link CBC pattern to mechanisms\n"
            "4. CRITICAL ALERTS: values requiring immediate action\n"
            "5. SEQUENTIAL INVESTIGATION PLAN: step-by-step with rationale\n"
            "6. COLLECTION QUALITY: any internal inconsistencies suggesting pre-analytical error"
        )
        return self.generate_with_rag(
            query=query, top_k=6,
            additional_context=f"Sex:{sex}, age:{age}. Abnormalities: {abnormals}",
        )


# ─────────────────────────────────────────────────────────
# KEYWORD FALLBACK RETRIEVAL  (no API required)
# ─────────────────────────────────────────────────────────

class KeywordRetriever:
    """
    TF-IDF-style keyword overlap retriever.
    Used when knowledge index is not yet built.
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

def create_rag_engine(kb_path: str, api_key: Optional[str] = None) -> CBCRagEngine:
    """Create and return a configured CBCRagEngine."""
    return CBCRagEngine(kb_path=kb_path, api_key=api_key)


def create_keyword_retriever(kb_path: str) -> KeywordRetriever:
    """Create keyword-based fallback retriever."""
    chunks = load_knowledge_base(kb_path)
    return KeywordRetriever(chunks)
