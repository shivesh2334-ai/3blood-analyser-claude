# 🧬 CBC RAG Analyzer

**Retrieval-Augmented Generation for Complete Blood Count Clinical Analysis**

A Streamlit application combining a curated UpToDate® CBC knowledge base with Google Gemini embeddings and generation for grounded, source-cited clinical interpretation.

---

## 🏗 RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CBC RAG PIPELINE                              │
│                                                                       │
│  KNOWLEDGE BASE          INDEXING              RETRIEVAL             │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────┐         │
│  │ 45 Clinical  │    │ Gemini         │    │ Cosine       │         │
│  │ CBC Chunks   │───▶│ text-embedding │───▶│ Similarity   │         │
│  │ (UpToDate®)  │    │ -004 API       │    │ Top-K Search │         │
│  └──────────────┘    └────────────────┘    └──────┬───────┘         │
│                                                    │                  │
│  USER INPUT            QUERY EMBEDDING            │ Retrieved Chunks │
│  ┌──────────────┐    ┌────────────────┐           │                  │
│  │ CBC Values   │───▶│ Gemini Embed   │           ▼                  │
│  │ Manual/OCR   │    │ RETRIEVAL_QUERY│    ┌──────────────┐         │
│  └──────────────┘    └────────────────┘    │ Context      │         │
│                                             │ Augmented    │         │
│  GENERATION          AUGMENTED PROMPT      │ Prompt       │         │
│  ┌──────────────┐    ┌────────────────┐    └──────┬───────┘         │
│  │ Gemini       │◀───│ Clinical Query │◀──────────┘                  │
│  │ 1.5-flash    │    │ + Retrieved    │                              │
│  │ (0.2 temp)   │    │   Knowledge    │                              │
│  └──────┬───────┘    └────────────────┘                              │
│         │                                                             │
│         ▼                                                             │
│  ┌──────────────┐                                                    │
│  │ Grounded     │  [Source 1] ... [Source N] cited inline            │
│  │ Clinical     │  Relevance scores shown                            │
│  │ Answer       │  Knowledge passages expandable                     │
│  └──────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Knowledge Base — 45 Clinical Chunks

| Section | Chunks | Topics |
|---------|--------|--------|
| Sample Collection & QC | 5 | Phlebotomy, EDTA, storage, Rule of Threes, MCHC QC |
| RBC Parameters | 6 | RBC/Hgb/HCT, MCV, RDW, MCH/MCHC, Reticulocytes, IRF |
| Platelet Parameters | 4 | PLT count, MPV, thrombocytopenia, thrombocytosis |
| WBC Parameters | 2 | WBC count, 5-part differential |
| Anemia Evaluation | 9 | MCV classification, IDA, thalassemia, macrocytic, normocytic, hemolysis, ACD, sickle cell, pregnancy anemia |
| Neutrophilia/penia | 4 | Mechanisms, reactive causes, CML, neutropenia causes/severity |
| Platelets (clinical) | 2 | Thrombocytopenia DDx, thrombocytosis DDx |
| Primary Immunodeficiency | 4 | CBC patterns, lymphopenia/SCID, antibody deficiency, phagocytic/NK disorders |
| Polycythemia | 1 | Classification, JAK2, workup |
| Eosinophilia | 1 | Causes, HES evaluation |
| Lymphocytosis/penia | 2 | CLL, reactive, acquired lymphopenia |
| Special Topics | 7 | Blood smear, MDS, aplastic anemia, ethnic neutropenia, hemoglobin variants, iron studies, critical values |

Each chunk contains:
- `section` — Clinical category
- `title` — Specific topic
- `keywords` — 8–15 search terms
- `text` — 150–300 words of clinical guideline content

---

## 🔍 Retrieval Strategy

**Embedding Model:** `models/text-embedding-004` (Gemini)
- Document embeddings: `task_type = RETRIEVAL_DOCUMENT`
- Query embeddings: `task_type = RETRIEVAL_QUERY`

**Similarity:** Cosine similarity (pure Python — no numpy required)

**Chunk Embedding:** Title + Section + Keywords + Full Text
```
"Section: Anemia Evaluation
 Title: Iron Deficiency Anemia - Diagnosis
 Keywords: iron deficiency, IDA, ferritin, transferrin...
 [Full clinical text]"
```

**Query Generation (auto):** Clinical questions are generated from CBC values:
```python
# Example for Hgb 9.0, MCV 70, RDW 18.5
query = """Patient: F, Hgb=9.0 g/dL, MCV=70 fL, RDW=18.5%, Reticulocytes=0.8%
Classify the type of anemia, identify most likely causes, explain pathophysiology,
and recommend specific next investigations. What does the RDW indicate?"""
```

**Top-K:** 4–6 chunks per query (configurable)

---

## 🚀 Deployment on Streamlit Community Cloud

### Repository Structure Required
```
your-repo/
├── app.py                    # Main Streamlit application
├── rag_engine.py             # RAG engine (embeddings + retrieval + generation)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   └── cbc_knowledge_base.json  # 45 pre-chunked knowledge entries
└── .streamlit/
    └── config.toml           # Dark theme configuration
```

### Steps
1. Fork/clone this repository
2. Push to your GitHub account
3. Go to [share.streamlit.io](https://share.streamlit.io) → New App
4. Select your repo, branch `main`, and file `app.py`
5. Click **Deploy**

> No Docker, no server configuration needed.

### Local Development
```bash
git clone https://github.com/YOUR_USERNAME/cbc-rag-analyzer
cd cbc-rag-analyzer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔑 API Key Setup

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

The API key powers:
1. **Embedding generation** (text-embedding-004) — only needed once per session to build the index
2. **CBC report OCR** (gemini-1.5-flash) — for PDF/image upload
3. **Clinical answer generation** (gemini-1.5-flash) — grounded in retrieved chunks

**Without API key:** The app runs in built-in logic mode with keyword-based knowledge retrieval.

---

## 🎛 Analysis Modes

| Mode | Description | API Key Required |
|------|-------------|-----------------|
| `🔧 Built-in Logic Only` | Rule-based algorithms + keyword knowledge search | ❌ No |
| `🧬 RAG Full Analysis` | Complete grounded narrative for all abnormalities | ✅ Yes |
| `🎯 RAG Targeted` | Per-section RAG (anemia / neutrophil / platelets etc.) | ✅ Yes |

---

## 🧪 RAG Chat Feature

Type any clinical question in the chat box:
- *"What does an elevated RDW with low MCV suggest?"*
- *"How do I differentiate ITP from TTP on a CBC?"*
- *"What are the CBC clues for SCID in an infant?"*
- *"When should I suspect CML vs leukemoid reaction?"*

The RAG engine will retrieve the most relevant knowledge passages and generate a grounded, cited answer.

---

## 📊 Technical Details

```python
# Cosine similarity (pure Python, no numpy)
def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    mag = sqrt(sum(x*x for x in a)) * sqrt(sum(x*x for x in b))
    return dot / mag

# Index building
for chunk in knowledge_base:
    text = f"Section: {chunk.section}\nTitle: {chunk.title}\n{chunk.text}"
    embedding = gemini.embed_content(model, text, task_type="RETRIEVAL_DOCUMENT")
    vector_store.add(chunk, embedding)

# Query time
query_emb = gemini.embed_content(model, clinical_query, task_type="RETRIEVAL_QUERY")
top_chunks = vector_store.search(query_emb, top_k=4)  # cosine similarity
context = format_chunks(top_chunks)
answer = gemini_flash.generate(f"Context: {context}\nQuestion: {query}")
```

---

## ⚠️ Disclaimer

This application is for **educational and clinical decision support purposes only**. It is not a substitute for clinical judgment. All interpretations should be validated by qualified healthcare professionals in the context of the complete clinical picture.

---

## 📄 License

MIT License — free for educational and non-commercial clinical use.

