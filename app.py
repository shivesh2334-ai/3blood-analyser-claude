"""
CBC Clinical Analyzer — RAG Edition  (Claude)
Retrieval-Augmented Generation · UpToDate Knowledge Base · Anthropic Claude

Architecture:
  Embeddings : sentence-transformers/all-MiniLM-L6-v2  (local, no API key)
  Generation : Anthropic Claude claude-3-5-haiku-20241022
  Retrieval  : Cosine similarity (pure Python)
"""

import streamlit as st
import json
import re
import os
import base64

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBC RAG Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(__file__), "data", "cbc_knowledge_base.json")

REFERENCE_RANGES = {
    "rbc_m":     (4.5,  5.9,   "×10¹²/L"),
    "rbc_f":     (4.0,  5.2,   "×10¹²/L"),
    "hgb_m":     (13.5, 17.5,  "g/dL"),
    "hgb_f":     (12.0, 15.5,  "g/dL"),
    "hct_m":     (41.0, 53.0,  "%"),
    "hct_f":     (36.0, 46.0,  "%"),
    "mcv":       (80.0, 100.0, "fL"),
    "mch":       (27.0, 33.0,  "pg"),
    "mchc":      (32.0, 36.0,  "g/dL"),
    "rdw":       (11.5, 14.5,  "%"),
    "retic":     (0.5,  2.5,   "%"),
    "wbc":       (4.5,  11.0,  "×10⁹/L"),
    "neut_abs":  (1.8,  7.7,   "×10⁹/L"),
    "neut_pct":  (40.0, 75.0,  "%"),
    "lymph_abs": (1.0,  4.8,   "×10⁹/L"),
    "lymph_pct": (20.0, 45.0,  "%"),
    "mono_abs":  (0.2,  1.0,   "×10⁹/L"),
    "eos_abs":   (0.0,  0.5,   "×10⁹/L"),
    "baso_abs":  (0.0,  0.1,   "×10⁹/L"),
    "plt":       (150.0,400.0, "×10⁹/L"),
    "mpv":       (7.5,  12.5,  "fL"),
}

CRITICAL_RANGES = {
    "hgb_m":    (7.0,  20.0),
    "hgb_f":    (7.0,  20.0),
    "plt":      (20.0, 1000.0),
    "wbc":      (1.5,  50.0),
    "neut_abs": (0.5,  None),
}

# ─────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:#0d1117; --surface:#161b22; --border:#30363d;
    --red:#f85149; --amber:#d29922; --green:#3fb950;
    --blue:#58a6ff; --purple:#bc8cff; --text:#e6edf3;
    --muted:#8b949e; --accent:#1f6feb;
    --claude:#da7756;
  }

  html,body,.stApp{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif}

  /* ── Header ── */
  .rag-header{
    background:linear-gradient(135deg,#0d1117 0%,#1a1a2e 50%,#0d1117 100%);
    border:1px solid var(--claude);border-radius:16px;
    padding:28px 36px;margin-bottom:28px;position:relative;overflow:hidden;
  }
  .rag-header::before{
    content:'';position:absolute;top:-50%;right:-20%;
    width:500px;height:500px;
    background:radial-gradient(circle,rgba(218,119,86,.08) 0%,transparent 60%);
    pointer-events:none;
  }
  .rag-header h1{font-family:'DM Serif Display',serif;font-size:2.2rem;color:#fff;margin:0}
  .rag-header .tagline{color:var(--claude);font-size:.9rem;margin-top:4px}
  .rag-badge{
    display:inline-flex;align-items:center;gap:6px;
    background:rgba(218,119,86,.12);border:1px solid var(--claude);
    color:var(--claude);padding:4px 12px;border-radius:20px;
    font-size:.75rem;margin-right:8px;margin-top:10px;
  }

  /* ── Step labels ── */
  .step-label{
    display:flex;align-items:center;gap:10px;
    font-size:.75rem;font-weight:600;letter-spacing:2px;
    text-transform:uppercase;color:var(--claude);
    margin:28px 0 12px;border-bottom:1px solid #21262d;padding-bottom:8px;
  }
  .step-num{
    background:var(--claude);color:#fff;border-radius:50%;
    width:22px;height:22px;display:flex;
    align-items:center;justify-content:center;font-size:.7rem;font-weight:700;
  }

  /* ── Parameter cards ── */
  .param-card{
    background:var(--surface);border:1px solid var(--border);
    border-radius:10px;padding:12px 16px;margin-bottom:8px;transition:all .15s;
  }
  .param-card:hover{border-color:var(--claude)}
  .param-card.low {border-left:3px solid var(--red);  background:rgba(248,81,73,.06)}
  .param-card.high{border-left:3px solid var(--amber);background:rgba(210,153,34,.06)}
  .param-card.crit{border-left:3px solid var(--purple);background:rgba(188,140,255,.1);animation:pulse-b 2s infinite}
  .param-card.ok  {border-left:3px solid var(--green)}
  @keyframes pulse-b{0%,100%{border-color:var(--purple)}50%{border-color:#ff79c6}}
  .param-value{font-size:1.6rem;font-weight:600;line-height:1;margin:4px 0 2px}
  .param-name {font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
  .param-ref  {font-size:.7rem;color:var(--muted)}
  .status-pill{display:inline-block;padding:1px 8px;border-radius:12px;font-size:.68rem;font-weight:600;margin-left:6px}
  .pill-ok  {background:rgba(63,185,80,.2);color:#3fb950}
  .pill-low {background:rgba(248,81,73,.2);color:#f85149}
  .pill-high{background:rgba(210,153,34,.2);color:#d29922}
  .pill-crit{background:rgba(188,140,255,.2);color:#bc8cff}

  /* ── Alert boxes ── */
  .alert{border-radius:8px;padding:12px 16px;margin:6px 0;font-size:.88rem;line-height:1.6}
  .alert-r{background:rgba(248,81,73,.1); border:1px solid rgba(248,81,73,.4)}
  .alert-a{background:rgba(210,153,34,.1);border:1px solid rgba(210,153,34,.4)}
  .alert-g{background:rgba(63,185,80,.1); border:1px solid rgba(63,185,80,.4)}
  .alert-b{background:rgba(88,166,255,.1);border:1px solid rgba(88,166,255,.4)}
  .alert-p{background:rgba(188,140,255,.1);border:1px solid rgba(188,140,255,.4)}

  /* ── RAG answer ── */
  .rag-answer{
    background:#0d1117;border:1px solid var(--claude);border-radius:12px;
    padding:20px 24px;font-size:.9rem;line-height:1.8;color:#e6edf3;position:relative;
  }
  .rag-answer::before{
    content:'◆ CLAUDE RAG ANALYSIS';font-size:.65rem;font-weight:700;letter-spacing:2px;
    color:var(--claude);display:block;margin-bottom:12px;
    border-bottom:1px solid #21262d;padding-bottom:8px;
  }

  /* ── Source cards ── */
  .source-card{background:#21262d;border:1px solid #30363d;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:.8rem}
  .source-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
  .source-title{color:var(--claude);font-weight:600}
  .source-score{color:var(--green);font-size:.72rem;font-weight:600}
  .source-section{color:var(--muted);font-size:.7rem}
  .source-preview{color:#8b949e;font-size:.75rem;line-height:1.5;margin-top:4px}

  /* ── Quality meter ── */
  .q-meter-bg{background:#21262d;border-radius:20px;height:10px;overflow:hidden;margin:6px 0}
  .q-meter-fill{height:100%;border-radius:20px;transition:width .5s ease}

  /* ── Upload zone ── */
  .upload-zone{
    background:rgba(218,119,86,.05);border:2px dashed var(--claude);
    border-radius:12px;padding:28px;text-align:center;color:var(--claude);font-size:.9rem;margin:8px 0;
  }

  /* ── Inputs ── */
  .stNumberInput input{
    background:#161b22!important;border:1px solid #30363d!important;
    color:#e6edf3!important;border-radius:8px!important;
    font-size:1rem!important;font-weight:500!important;
  }
  .stNumberInput input:focus{border-color:var(--claude)!important}

  /* ── Button ── */
  .stButton>button{
    background:linear-gradient(135deg,#da7756 0%,#b85a3a 100%)!important;
    color:white!important;border:none!important;border-radius:10px!important;
    font-weight:600!important;font-size:1rem!important;
    padding:12px 32px!important;transition:all .2s!important;
  }
  .stButton>button:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(218,119,86,.4)!important}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"]{gap:2px;background:#161b22!important;border-radius:10px;padding:4px}
  .stTabs [data-baseweb="tab"]     {border-radius:8px!important;color:#8b949e!important;font-size:.85rem!important}
  .stTabs [aria-selected="true"]   {background:var(--claude)!important;color:#fff!important}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d}
  section[data-testid="stSidebar"] *{color:#e6edf3!important}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar{width:6px}
  ::-webkit-scrollbar-track{background:#161b22}
  ::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}

  /* ── Index status bar ── */
  .index-ready{
    background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.4);
    border-radius:8px;padding:8px 14px;font-size:.8rem;color:#3fb950;margin:8px 0;
  }
  .index-building{
    background:rgba(218,119,86,.1);border:1px solid rgba(218,119,86,.4);
    border-radius:8px;padding:8px 14px;font-size:.8rem;color:var(--claude);margin:8px 0;
  }

  /* ── Footer ── */
  .footer{
    background:#161b22;border:1px solid #30363d;border-radius:10px;
    padding:10px 18px;text-align:center;color:#8b949e;font-size:.75rem;margin-top:28px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────

def nz(v):
    return v if v else None


def step_label(num, text):
    st.markdown(
        f'<div class="step-label"><div class="step-num">{num}</div>{text}</div>',
        unsafe_allow_html=True,
    )


def classify_value(value, lo, hi, crit_lo=None, crit_hi=None):
    if value is None:               return "unknown"
    if crit_lo and value < crit_lo: return "crit_low"
    if crit_hi and value > crit_hi: return "crit_high"
    if value < lo:                  return "low"
    if value > hi:                  return "high"
    return "ok"


def render_param_card(label, value, unit, lo, hi, crit_lo=None, crit_hi=None):
    if value is None:
        return
    status    = classify_value(value, lo, hi, crit_lo, crit_hi)
    pill_cls  = {"ok":"pill-ok","low":"pill-low","high":"pill-high",
                 "crit_low":"pill-crit","crit_high":"pill-crit"}.get(status,"")
    card_cls  = {"ok":"ok","low":"low","high":"high",
                 "crit_low":"crit","crit_high":"crit"}.get(status,"")
    pill_txt  = {"ok":"✓ Normal","low":"↓ Low","high":"↑ High",
                 "crit_low":"⚠ Critical","crit_high":"⚠ Critical"}.get(status,"")
    val_color = {"ok":"#3fb950","low":"#f85149","high":"#d29922",
                 "crit_low":"#bc8cff","crit_high":"#bc8cff"}.get(status,"#e6edf3")
    st.markdown(f"""
    <div class="param-card {card_cls}">
      <div class="param-name">{label}</div>
      <div class="param-value" style="color:{val_color}">{value:.2f}
        <span style="font-size:.8rem;color:#8b949e"> {unit}</span>
        <span class="status-pill {pill_cls}">{pill_txt}</span>
      </div>
      <div class="param-ref">Ref: {lo}–{hi} {unit}</div>
    </div>""", unsafe_allow_html=True)


def render_alert(text, kind="b"):
    text_html = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#e6edf3">\1</strong>', text)
    text_html = text_html.replace('\n', '<br>')
    cls = {"r":"alert-r","a":"alert-a","g":"alert-g","b":"alert-b","p":"alert-p"}.get(kind,"alert-b")
    st.markdown(f'<div class="alert {cls}">{text_html}</div>', unsafe_allow_html=True)


def render_rag_answer(result: dict):
    if not result:
        return
    answer  = result.get("answer", "")
    sources = result.get("sources", [])
    st.markdown(f'<div class="rag-answer">{answer}</div>', unsafe_allow_html=True)
    if sources:
        with st.expander(f"📚 {len(sources)} Knowledge Sources Retrieved", expanded=False):
            for src in sources:
                pct   = int(src["score"] * 100)
                color = "#3fb950" if pct > 70 else ("#d29922" if pct > 50 else "#8b949e")
                st.markdown(f"""
                <div class="source-card">
                  <div class="source-header">
                    <div>
                      <span class="source-title">[Source {src['index']}] {src['title']}</span><br>
                      <span class="source-section">{src['section']}</span>
                    </div>
                    <span class="source-score">Relevance: {pct}%</span>
                  </div>
                  <div style="background:#161b22;border-radius:4px;height:4px;overflow:hidden;margin:4px 0">
                    <div style="background:{color};height:100%;width:{pct}%"></div>
                  </div>
                  <div class="source-preview">{src['preview']}</div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SAMPLE QUALITY
# ─────────────────────────────────────────────────────────

def rule_of_threes(rbc, hgb, hct):
    issues = []
    if rbc and hgb:
        exp = rbc * 3
        if exp > 0 and abs(hgb - exp) / exp > 0.10:
            issues.append(f"Hgb ({hgb:.1f}) deviates >{abs(hgb-exp)/exp*100:.0f}% from RBC×3 ({exp:.1f})")
    if hgb and hct:
        exp = hgb * 3
        if exp > 0 and abs(hct - exp) / exp > 0.10:
            issues.append(f"HCT ({hct:.1f}) deviates >{abs(hct-exp)/exp*100:.0f}% from Hgb×3 ({exp:.1f})")
    return issues


def sample_quality(data):
    issues = []; warnings = []; score = 100
    for r in rule_of_threes(data.get("rbc"), data.get("hgb"), data.get("hct")):
        issues.append(f"⚠ Rule-of-Threes: {r}"); score -= 20
    mchc = data.get("mchc")
    if mchc and mchc > 36:
        issues.append("⚠ MCHC >36 g/dL — hemolysis, cold agglutinin, lipemia, or very high WBC"); score -= 15
    if mchc and mchc < 28:
        warnings.append("ℹ Very low MCHC — confirm no dilution or pre-analytical error")
    plt = data.get("plt"); wbc = data.get("wbc")
    if plt and plt < 100 and wbc and wbc > 12:
        warnings.append("ℹ Low PLT + leukocytosis — consider pseudothrombocytopenia from platelet clumping")
    for k in ["rbc", "hgb", "hct", "wbc", "plt"]:
        if data.get(k) is not None and data[k] <= 0:
            issues.append(f"⚠ {k.upper()} ≤ 0 — likely data entry error"); score -= 30
    return max(0, min(100, score)), issues, warnings


# ─────────────────────────────────────────────────────────
# BUILT-IN CLINICAL LOGIC  (no API required)
# ─────────────────────────────────────────────────────────

def built_in_anemia(data, sex):
    hgb = data.get("hgb"); mcv = data.get("mcv"); rdw = data.get("rdw")
    retic = data.get("retic"); hct = data.get("hct")
    hgb_lo = 13.5 if sex == "M" else 12.0
    out = []
    if not hgb or hgb >= hgb_lo:
        if hgb: out.append(("g", "✅ No anemia detected"))
        return out
    sev = "Mild" if hgb >= 10 else ("Moderate" if hgb >= 8 else "Severe")
    out.append(("r", f"🔴 **Anemia** — Hgb {hgb:.1f} g/dL [{sev}]"))
    if mcv:
        if mcv < 80:
            out.append(("a", "📊 **Microcytic** (MCV <80 fL)"))
            if rdw and rdw > 14.5:
                out.append(("b", "→ High RDW + microcytosis → **Iron Deficiency Anemia** most likely. Check ferritin, serum iron, TIBC."))
            else:
                out.append(("b", "→ Normal RDW + microcytosis → **Thalassemia trait** or ACD. Check HbA2 electrophoresis."))
        elif mcv > 100:
            out.append(("a", "📊 **Macrocytic** (MCV >100 fL)"))
            out.append(("b", "→ Evaluate: B12, Folate, TSH, LFTs, medications, blood smear for hypersegmented neutrophils."))
        else:
            out.append(("a", "📊 **Normocytic** (MCV 80–100 fL)"))
            if retic and retic > 2.5:
                out.append(("b", "→ Elevated retics → **Hemolytic** or blood loss recovery. Check LDH, haptoglobin, Coombs."))
            else:
                out.append(("b", "→ Low retics → **Hypoproliferative** (ACD, renal disease, aplasia). Check creatinine, ferritin."))
    if retic and hct:
        mat = 1.0 if hct >= 35 else (1.5 if hct >= 25 else 2.0)
        rpi = (retic / 100 * hct / 45) / mat * 100
        out.append(("g" if rpi >= 2 else "a", f"→ **RPI = {rpi:.1f}** — {'Adequate marrow response' if rpi>=2 else 'Inadequate marrow response'}"))
    return out


def built_in_neutrophil(data):
    wbc = data.get("wbc"); neut_abs = data.get("neut_abs")
    neut_pct = data.get("neut_pct"); bands = data.get("bands")
    anc = neut_abs or (wbc * neut_pct / 100 if wbc and neut_pct else None)
    out = []
    if anc is None: return out
    if anc > 7.7:
        out.append(("r", f"🔴 **Neutrophilia** — ANC {anc:.2f} ×10⁹/L"))
        if anc >= 100:
            out.append(("r", "⚠ Extreme leukocytosis — BCR-ABL for CML, leukapheresis if symptomatic"))
        elif anc >= 50:
            out.append(("a", "→ Leukemoid reaction vs MPN. BCR-ABL, LAP score, bone marrow biopsy."))
        else:
            out.append(("b", "→ Likely reactive: infection, corticosteroids, smoking, stress, post-splenectomy"))
        if bands and bands > 5:
            out.append(("a", f"→ Left shift (Bands {bands:.0f}%) — consistent with active infection/stress"))
    elif anc < 1.8:
        out.append(("r", f"🔴 **Neutropenia** — ANC {anc:.2f} ×10⁹/L"))
        if anc < 0.5:
            out.append(("r", "⚠ **Severe neutropenia** — high infection risk. Blood cultures if febrile; empiric antibiotics."))
        out.append(("b", "→ Check: medication list, viral titers (EBV/CMV/HIV), ANA, B12/folate, bone marrow if unexplained"))
    return out


def built_in_platelets(data):
    plt = data.get("plt"); mpv = data.get("mpv"); out = []
    if not plt: return out
    if plt < 150:
        out.append(("r", f"🔴 **Thrombocytopenia** — PLT {plt:.0f} ×10⁹/L"))
        if plt < 20:
            out.append(("r", "⚠ PLT <20 — spontaneous bleeding risk. Urgent haematology review."))
        if mpv:
            if mpv > 12.5:
                out.append(("b", "→ High MPV → active megakaryopoiesis (ITP most likely)"))
            elif mpv < 7.5:
                out.append(("b", "→ Low MPV → marrow suppression (aplastic anaemia, Wiskott-Aldrich)"))
        out.append(("b", "→ First step: rule out EDTA pseudothrombocytopenia — repeat in citrate tube, smear for aggregates"))
    elif plt > 400:
        out.append(("a", f"🔸 **Thrombocytosis** — PLT {plt:.0f} ×10⁹/L"))
        if plt > 1000:
            out.append(("r", "→ Extreme thrombocytosis — JAK2 V617F, CALR, MPL mutation testing for ET/PV/CML"))
        else:
            out.append(("b", "→ Consider reactive: iron deficiency, inflammation, infection, post-splenectomy, malignancy"))
    return out


# ─────────────────────────────────────────────────────────
# PDF / IMAGE OCR via Claude Vision
# ─────────────────────────────────────────────────────────

def pdf_to_image_bytes(pdf_bytes):
    try:
        import fitz
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        pix  = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return pix.tobytes("png"), "image/png"
    except ImportError:
        return None, None


def extract_cbc_with_claude(api_key: str, img_bytes: bytes, mime_type: str) -> dict:
    """
    Uses Claude (claude-3-5-haiku) vision to extract CBC values from a lab report image.
    """
    import anthropic

    client     = anthropic.Anthropic(api_key=api_key)
    image_data = base64.standard_b64encode(img_bytes).decode("utf-8")

    prompt = """Extract all NUMERIC CBC values from this lab report image.

CRITICAL INSTRUCTIONS:
- Look for the main CBC results table with numeric values (usually in the first page)
- Extract ONLY numbers - strip all units (g/dL, ×10⁹/L, %, fL, etc.)
- If a value is not found or is qualitative text (e.g. "Normal", "Adequate"), use null
- Ignore peripheral smear interpretations and qualitative comments
- Common parameter names: Hemoglobin/Hb/Hgb, RBC/Red Cell Count, WBC/White Cell Count, 
  Platelets/PLT, MCV, MCH, MCHC, Neutrophils/Neut, Lymphocytes/Lymph, etc.

Return ONLY valid JSON (no markdown, no explanation):
{"rbc":null,"hgb":null,"hct":null,"mcv":null,"mch":null,"mchc":null,"rdw":null,
"retic":null,"wbc":null,"neut_abs":null,"neut_pct":null,"lymph_abs":null,"lymph_pct":null,
"mono_abs":null,"mono_pct":null,"eos_abs":null,"eos_pct":null,"baso_abs":null,"baso_pct":null,
"bands":null,"plt":null,"mpv":null,"immature_gran":null,"nrbc":null}

EXAMPLES:
Report shows "Hemoglobin: 12.5 g/dL" → "hgb":12.5
Report shows "WBC: 8.2 ×10⁹/L" → "wbc":8.2
Report shows "Neutrophils: 65%" → "neut_pct":65
Report shows "Platelets: Adequate" → "plt":null
"""

    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": mime_type,
                        "data":       image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    text  = message.content[0].text
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ─────────────────────────────────────────────────────────
# SESSION-STATE RAG ENGINE CACHE
# ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _build_engine_cached(kb_path: str):
    """
    Builds the vector index once per server lifecycle using local sentence-transformers.
    Cached across all users/reruns — no API key needed for this step.
    """
    from rag_engine import CBCRagEngine
    engine = CBCRagEngine(kb_path=kb_path)
    engine.build_index()
    return engine


def get_rag_engine(api_key: str = None):
    """
    Returns the cached RAG engine, injecting the API key for generation.
    Index is built once locally (no API key needed).
    """
    engine         = _build_engine_cached(KB_PATH)
    engine.api_key = api_key   # update key for generation calls
    return engine


def get_keyword_retriever():
    if "kw_retriever" not in st.session_state:
        from rag_engine import create_keyword_retriever
        st.session_state["kw_retriever"] = create_keyword_retriever(KB_PATH)
    return st.session_state["kw_retriever"]


# ─────────────────────────────────────────────────────────
# INDEX STATUS WIDGET
# ─────────────────────────────────────────────────────────

def show_index_status():
    """Shows the current knowledge index build status in the sidebar."""
    try:
        engine = _build_engine_cached(KB_PATH)
        if engine.is_ready():
            n = len(engine.store)
            st.markdown(
                f'<div class="index-ready">✅ Knowledge index ready — {n} chunks loaded</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="index-building">⏳ Building knowledge index…</div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.warning(f"Index status unknown: {e}")


# ─────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────

def main():

    # ── HEADER ────────────────────────────────────────────
    st.markdown("""
    <div class="rag-header">
      <h1>🧬 CBC RAG Analyzer</h1>
      <div class="tagline">Retrieval-Augmented Generation · UpToDate Knowledge Base · Claude AI</div>
      <div>
        <span class="rag-badge">◆ 51 Knowledge Chunks</span>
        <span class="rag-badge">🔍 Semantic Retrieval</span>
        <span class="rag-badge">📖 Source-Cited Answers</span>
        <span class="rag-badge">🧠 Local Embeddings</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        sex = st.selectbox("Patient Sex", ["M", "F"],
                           format_func=lambda x: "🧑 Male" if x == "M" else "👩 Female")
        age = st.number_input("Age (years)", 0, 120, 35, 1)

        st.divider()
        st.markdown("### 🤖 Claude AI")
        api_key = st.text_input(
            "Anthropic API Key", type="password",
            help="Get your key at console.anthropic.com — used for generation & OCR only"
        )

        if api_key:
            st.markdown(
                '<div style="font-size:.75rem;color:#3fb950">✅ API key entered</div>',
                unsafe_allow_html=True
            )

        rag_mode = st.selectbox(
            "Analysis Mode",
            ["built_in", "rag_full", "rag_targeted"],
            format_func=lambda x: {
                "built_in":     "🔧 Built-in Logic Only",
                "rag_full":     "◆ RAG Full Analysis",
                "rag_targeted": "🎯 RAG Targeted (per section)",
            }[x],
            help="RAG modes require an Anthropic API key for generation"
        )

        st.divider()
        st.markdown("### 📚 Knowledge Index")
        show_index_status()
        st.markdown("""
        <div style="font-size:.72rem;color:#8b949e;line-height:1.7;margin-top:6px">
        Index builds automatically using local sentence-transformers.<br>
        No API key needed for indexing — only for Claude generation.
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style="font-size:.75rem;color:#8b949e;line-height:1.7">
        <b>RAG Architecture:</b><br>
        1. 51 CBC guideline chunks (UpToDate)<br>
        2. all-MiniLM-L6-v2 (local embeddings)<br>
        3. Cosine similarity retrieval<br>
        4. Claude claude-3-5-haiku generation<br>
        5. Source citations in every response
        </div>""", unsafe_allow_html=True)

    # ── DATA INPUT TABS ───────────────────────────────────
    step_label("1", "Enter CBC Values")
    in_tab, upload_tab = st.tabs(["✏️ Manual Entry", "📄 Upload Report (PDF/Image)"])
    data = {}

    with in_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div style="color:#f85149;font-weight:600;margin-bottom:8px">🔴 Red Blood Cell Series</div>', unsafe_allow_html=True)
            data["rbc"]   = nz(st.number_input("RBC (×10¹²/L)",        0.0, 15.0,  0.0, 0.01,  "%.2f"))
            data["hgb"]   = nz(st.number_input("Hemoglobin (g/dL)",     0.0, 25.0,  0.0, 0.1,   "%.1f"))
            data["hct"]   = nz(st.number_input("Hematocrit (%)",        0.0, 75.0,  0.0, 0.1,   "%.1f"))
            data["mcv"]   = nz(st.number_input("MCV (fL)",              0.0, 200.0, 0.0, 0.1,   "%.1f"))
            data["mch"]   = nz(st.number_input("MCH (pg)",              0.0, 60.0,  0.0, 0.1,   "%.1f"))
            data["mchc"]  = nz(st.number_input("MCHC (g/dL)",          0.0, 50.0,  0.0, 0.1,   "%.1f"))
            data["rdw"]   = nz(st.number_input("RDW (%)",               0.0, 30.0,  0.0, 0.1,   "%.1f"))
            data["retic"] = nz(st.number_input("Reticulocytes (%)",     0.0, 20.0,  0.0, 0.01,  "%.2f"))
        with c2:
            st.markdown('<div style="color:#58a6ff;font-weight:600;margin-bottom:8px">⚪ White Blood Cell Series</div>', unsafe_allow_html=True)
            data["wbc"]       = nz(st.number_input("WBC (×10⁹/L)",              0.0, 500.0, 0.0, 0.1,   "%.1f"))
            data["neut_abs"]  = nz(st.number_input("Neutrophils Abs (×10⁹/L)",  0.0, 300.0, 0.0, 0.01,  "%.2f"))
            data["neut_pct"]  = nz(st.number_input("Neutrophils (%)",            0.0, 100.0, 0.0, 0.1,   "%.1f"))
            data["lymph_abs"] = nz(st.number_input("Lymphocytes Abs (×10⁹/L)", 0.0, 200.0, 0.0, 0.01,  "%.2f"))
            data["lymph_pct"] = nz(st.number_input("Lymphocytes (%)",           0.0, 100.0, 0.0, 0.1,   "%.1f"))
            data["mono_abs"]  = nz(st.number_input("Monocytes Abs (×10⁹/L)",   0.0, 50.0,  0.0, 0.01,  "%.2f"))
            data["eos_abs"]   = nz(st.number_input("Eosinophils Abs (×10⁹/L)", 0.0, 50.0,  0.0, 0.001, "%.3f"))
            data["baso_abs"]  = nz(st.number_input("Basophils Abs (×10⁹/L)",   0.0, 10.0,  0.0, 0.001, "%.3f"))
            data["bands"]     = nz(st.number_input("Bands (%)",                 0.0, 100.0, 0.0, 0.1,   "%.1f"))
        with c3:
            st.markdown('<div style="color:#d29922;font-weight:600;margin-bottom:8px">🟡 Platelet Series</div>', unsafe_allow_html=True)
            data["plt"] = nz(st.number_input("Platelets (×10⁹/L)", 0.0, 3000.0, 0.0, 1.0, "%.0f"))
            data["mpv"] = nz(st.number_input("MPV (fL)",            0.0, 30.0,   0.0, 0.1, "%.1f"))
            st.divider()
            st.markdown('<div style="color:#bc8cff;font-weight:600;margin-bottom:8px">🔬 Extended Parameters</div>', unsafe_allow_html=True)
            data["immature_gran"] = nz(st.number_input("Immature Granulocytes (%)", 0.0, 100.0, 0.0, 0.1, "%.1f"))
            data["nrbc"]          = nz(st.number_input("Nucleated RBCs (%)",        0.0, 100.0, 0.0, 0.1, "%.1f"))
            data["mono_pct"]      = nz(st.number_input("Monocytes (%)",             0.0, 100.0, 0.0, 0.1, "%.1f"))
            data["eos_pct"]       = nz(st.number_input("Eosinophils (%)",           0.0, 100.0, 0.0, 0.1, "%.1f"))
            data["baso_pct"]      = nz(st.number_input("Basophils (%)",             0.0, 100.0, 0.0, 0.1, "%.1f"))

    with upload_tab:
        if not api_key:
            st.markdown(
                '<div class="upload-zone">🔑 Enter Anthropic API key in sidebar to enable report OCR via Claude Vision</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="upload-zone">📄 Upload CBC report — PDF, JPG, PNG or JPEG<br>'
                'Claude Vision will extract values automatically</div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader("", type=["pdf", "jpg", "jpeg", "png"],
                                        label_visibility="collapsed")
            if uploaded:
                with st.spinner("🔍 Claude Vision extracting values…"):
                    try:
                        file_bytes = uploaded.read()
                        if uploaded.type == "application/pdf":
                            img_bytes, mime = pdf_to_image_bytes(file_bytes)
                            if img_bytes is None:
                                st.error("PDF parsing requires PyMuPDF. Install: pip install PyMuPDF")
                                img_bytes = None
                        else:
                            img_bytes, mime = file_bytes, uploaded.type

                        if img_bytes:
                            st.image(img_bytes, caption="Uploaded Report",
                                     use_container_width=True)
                            extracted = extract_cbc_with_claude(api_key, img_bytes, mime)
                            if extracted:
                                n = sum(1 for v in extracted.values() if v is not None)
                                if n == 0:
                                    st.warning(
                                        "⚠️ No numeric CBC values found. This may be:\n"
                                        "• A peripheral smear interpretation page (qualitative only)\n"
                                        "• A page without the main CBC results table\n"
                                        "• An unsupported report format\n\n"
                                        "**Try uploading the page with numeric values** "
                                        "(Hemoglobin, WBC count, Platelet count, etc.)"
                                    )
                                else:
                                    for k, v in extracted.items():
                                        if v is not None and k in data:
                                            data[k] = float(v)
                                    st.success(f"✅ Claude extracted {n} parameters")
                                    st.json({k: v for k, v in extracted.items() if v is not None})
                    except Exception as e:
                        st.error(f"Extraction error: {e}")

    # ── RAG CHAT ─────────────────────────────────────────
    custom_q    = None
    ask_clicked = False
    if rag_mode != "built_in" and api_key:
        st.divider()
        step_label("2", "Ask a Clinical Question (RAG Chat)")
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            custom_q = st.text_input(
                "", label_visibility="collapsed",
                placeholder="e.g. 'What does elevated RDW with low MCV suggest?' or 'How to differentiate ITP from TTP?'"
            )
        with col_btn:
            ask_clicked = st.button("Ask →", use_container_width=True)

    # ── ANALYZE BUTTON ────────────────────────────────────
    st.divider()
    analyze_clicked = st.button("🔬 Run Complete CBC Analysis",
                                type="primary", use_container_width=True)

    if not analyze_clicked and not ask_clicked:
        st.markdown("""
        <div class="alert alert-b" style="text-align:center;padding:20px">
          Fill in CBC values above, then click <strong>Run Complete CBC Analysis</strong><br>
          <span style="font-size:.8rem;color:#8b949e">
            RAG modes use Claude for generation — enter Anthropic API key in sidebar<br>
            Built-in Logic mode works without any API key
          </span>
        </div>""", unsafe_allow_html=True)
        st.markdown(
            '<div class="footer">🧬 CBC RAG Analyzer · UpToDate Knowledge Base · '
            'all-MiniLM-L6-v2 Local Embeddings · Claude claude-3-5-haiku · For clinical decision support only</div>',
            unsafe_allow_html=True,
        )
        return

    entered = {k: v for k, v in data.items() if v is not None and v > 0}
    if not entered and analyze_clicked:
        st.error("⚠️ Please enter at least some CBC values.")
        return

    # ── RAG CHAT RESPONSE ─────────────────────────────────
    if ask_clicked and custom_q and api_key:
        step_label("●", "Claude RAG Answer")
        with st.spinner("◆ Retrieving knowledge & generating answer with Claude…"):
            try:
                engine = get_rag_engine(api_key)
                result = engine.generate_with_rag(
                    query=custom_q, top_k=4,
                    additional_context=(f"CBC context: {json.dumps(entered)}" if entered else "")
                )
                render_rag_answer(result)
            except Exception as e:
                st.error(f"RAG error: {e}")
        return

    # ═══════════════════════════════════════════════════════
    # FULL ANALYSIS OUTPUT
    # ═══════════════════════════════════════════════════════
    hgb_lo, hgb_hi = (13.5, 17.5) if sex == "M" else (12.0, 15.5)
    rbc_lo, rbc_hi = (4.5, 5.9)   if sex == "M" else (4.0, 5.2)
    hct_lo, hct_hi = (41.0, 53.0) if sex == "M" else (36.0, 46.0)
    hgb_crit       = CRITICAL_RANGES["hgb_m"]

    # ── 1. PARAMETER REVIEW ───────────────────────────────
    step_label("1", "CBC Parameter Review")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown('<div style="color:#f85149;font-size:.8rem;font-weight:700;margin-bottom:8px;letter-spacing:1px">RBC SERIES</div>', unsafe_allow_html=True)
        render_param_card("RBC",           data.get("rbc"),       "×10¹²/L", rbc_lo, rbc_hi)
        render_param_card("Hemoglobin",    data.get("hgb"),       "g/dL",    hgb_lo, hgb_hi, hgb_crit[0], hgb_crit[1])
        render_param_card("Hematocrit",    data.get("hct"),       "%",       hct_lo, hct_hi)
        render_param_card("MCV",           data.get("mcv"),       "fL",      *REFERENCE_RANGES["mcv"][:2])
        render_param_card("MCH",           data.get("mch"),       "pg",      *REFERENCE_RANGES["mch"][:2])
        render_param_card("MCHC",          data.get("mchc"),      "g/dL",    *REFERENCE_RANGES["mchc"][:2])
        render_param_card("RDW",           data.get("rdw"),       "%",       *REFERENCE_RANGES["rdw"][:2])
        render_param_card("Reticulocytes", data.get("retic"),     "%",       *REFERENCE_RANGES["retic"][:2])
    with pc2:
        st.markdown('<div style="color:#58a6ff;font-size:.8rem;font-weight:700;margin-bottom:8px;letter-spacing:1px">WBC SERIES</div>', unsafe_allow_html=True)
        render_param_card("WBC",             data.get("wbc"),       "×10⁹/L", *REFERENCE_RANGES["wbc"][:2],      *CRITICAL_RANGES["wbc"])
        render_param_card("Neutrophils Abs", data.get("neut_abs"),  "×10⁹/L", *REFERENCE_RANGES["neut_abs"][:2], CRITICAL_RANGES["neut_abs"][0], None)
        render_param_card("Neutrophils %",   data.get("neut_pct"),  "%",       *REFERENCE_RANGES["neut_pct"][:2])
        render_param_card("Lymphocytes Abs", data.get("lymph_abs"), "×10⁹/L", *REFERENCE_RANGES["lymph_abs"][:2])
        render_param_card("Lymphocytes %",   data.get("lymph_pct"), "%",       *REFERENCE_RANGES["lymph_pct"][:2])
        render_param_card("Monocytes Abs",   data.get("mono_abs"),  "×10⁹/L", *REFERENCE_RANGES["mono_abs"][:2])
        render_param_card("Eosinophils Abs", data.get("eos_abs"),   "×10⁹/L", *REFERENCE_RANGES["eos_abs"][:2])
        render_param_card("Basophils Abs",   data.get("baso_abs"),  "×10⁹/L", *REFERENCE_RANGES["baso_abs"][:2])
    with pc3:
        st.markdown('<div style="color:#d29922;font-size:.8rem;font-weight:700;margin-bottom:8px;letter-spacing:1px">PLATELET SERIES</div>', unsafe_allow_html=True)
        render_param_card("Platelets", data.get("plt"), "×10⁹/L", *REFERENCE_RANGES["plt"][:2], *CRITICAL_RANGES["plt"])
        render_param_card("MPV",       data.get("mpv"), "fL",     *REFERENCE_RANGES["mpv"][:2])
        if data.get("immature_gran"):
            render_param_card("Immature Granulocytes", data["immature_gran"], "%", 0, 0)
        if data.get("nrbc"):
            render_param_card("Nucleated RBCs", data["nrbc"], "%", 0, 0)

    # ── 2. SAMPLE QUALITY ─────────────────────────────────
    step_label("2", "Sample Quality Assessment")
    q_score, q_issues, q_warns = sample_quality(data)
    fill = "#3fb950" if q_score >= 80 else ("#d29922" if q_score >= 50 else "#f85149")
    qlbl = "✅ Good Quality" if q_score >= 80 else ("⚠ Questionable" if q_score >= 50 else "🔴 Poor Quality")
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <strong>Sample Quality: {q_score}/100 — {qlbl}</strong>
        <span style="color:{fill};font-size:1.3rem;font-weight:700">{q_score}%</span>
      </div>
      <div class="q-meter-bg"><div class="q-meter-fill" style="width:{q_score}%;background:{fill}"></div></div>
      <div style="font-size:.75rem;color:#8b949e">Rule-of-Threes · MCHC sanity · Cross-parameter consistency</div>
    </div>""", unsafe_allow_html=True)
    for iss in q_issues: render_alert(iss, "r")
    for w   in q_warns:  render_alert(w,   "a")
    if not q_issues and not q_warns:
        render_alert("✅ No sample quality issues detected.", "g")

    # ── 3. SEQUENTIAL ANALYSIS TABS ───────────────────────
    step_label("3", "Sequential Clinical Analysis")
    tabs = st.tabs(["🩸 Anemia", "⚪ Neutrophil", "🟡 Platelets",
                    "🛡 Immunodeficiency", "🔴 Polycythemia", "🔬 Other Findings"])

    def rag_or_builtin(rag_fn, builtin_fn, *args):
        if rag_mode == "built_in" or not api_key:
            items = builtin_fn(*args)
            if not items: render_alert("✅ Within normal range.", "g")
            for k, v in items: render_alert(v, k)
            return
        with st.spinner("◆ Claude is retrieving & analysing…"):
            try:
                engine = get_rag_engine(api_key)
                result = rag_fn(engine)
                if result: render_rag_answer(result)
                else:      render_alert("✅ Within normal range.", "g")
            except Exception as e:
                st.error(f"Claude RAG error: {e}")
                items = builtin_fn(*args)
                for k, v in items: render_alert(v, k)

    with tabs[0]:
        st.markdown("#### Anemia Evaluation")
        rag_or_builtin(lambda eng: eng.analyze_anemia(data, sex), built_in_anemia, data, sex)

    with tabs[1]:
        st.markdown("#### Neutrophil Evaluation")
        rag_or_builtin(lambda eng: eng.analyze_neutrophil_abnormality(data), built_in_neutrophil, data)

    with tabs[2]:
        st.markdown("#### Platelet Evaluation")
        rag_or_builtin(lambda eng: eng.analyze_platelet_abnormality(data), built_in_platelets, data)

    with tabs[3]:
        st.markdown("#### Primary Immunodeficiency Screening")
        wbc_v   = data.get("wbc", 0) or 0
        lymph_v = data.get("lymph_abs") or (wbc_v * (data.get("lymph_pct") or 0) / 100)
        neut_v  = data.get("neut_abs")
        plt_v   = data.get("plt")
        mpv_v   = data.get("mpv")
        pid_flags = []
        if lymph_v and lymph_v < 1.0: pid_flags.append(f"Lymphopenia (ALC {lymph_v:.2f})")
        if neut_v  and neut_v  < 0.5: pid_flags.append(f"Severe neutropenia (ANC {neut_v:.2f})")
        if plt_v and plt_v < 100 and mpv_v and mpv_v < 7.5 and sex == "M":
            pid_flags.append("Low PLT + low MPV in male (Wiskott-Aldrich pattern)")

        if rag_mode == "built_in" or not api_key:
            if pid_flags:
                for f in pid_flags: render_alert(f"🛡 PID Flag: {f}", "p")
                render_alert("→ Evaluate: lymphocyte subsets (CD3/4/8/19/NK), immunoglobulins (IgG/A/M/E), vaccine titers.", "b")
            else:
                render_alert("✅ No CBC-based PID flags. Note: antibody deficiencies (CVID, XLA) can present with a normal CBC.", "g")
        elif pid_flags:
            with st.spinner("◆ Claude RAG: immunodeficiency guidelines…"):
                try:
                    engine = get_rag_engine(api_key)
                    render_rag_answer(engine.analyze_immunodeficiency_risk(data, sex, age))
                except Exception as e:
                    st.error(f"Claude RAG error: {e}")
                    for f in pid_flags: render_alert(f"🛡 PID Flag: {f}", "p")
        else:
            render_alert("✅ No CBC-based PID flags identified.", "g")

    with tabs[4]:
        st.markdown("#### Erythrocytosis / Polycythemia")
        hgb_hi2 = 17.5 if sex == "M" else 15.5
        hgb_v   = data.get("hgb")
        if hgb_v and hgb_v > hgb_hi2:
            render_alert(f"🔴 **Erythrocytosis** — Hgb {hgb_v:.1f} g/dL (>{hgb_hi2} g/dL)", "r")
            render_alert(
                "→ Distinguish relative (dehydration) from absolute erythrocytosis.\n"
                "→ Check: O₂ saturation, serum EPO (low→PV; high→secondary), JAK2 V617F.\n"
                "→ Secondary: COPD, sleep apnoea, high altitude, EPO-secreting tumour.\n"
                "→ Primary PV: JAK2+ + panmyelosis; bone marrow biopsy for confirmation.", "b"
            )
            if rag_mode != "built_in" and api_key:
                with st.spinner("◆ Claude RAG: polycythaemia guidelines…"):
                    try:
                        engine = get_rag_engine(api_key)
                        result = engine.generate_with_rag(
                            f"Patient has erythrocytosis: Hgb {hgb_v} g/dL ({sex}). Classify and provide workup.",
                            top_k=3,
                            additional_context=f"Sex:{sex}, CBC:{json.dumps(entered)}"
                        )
                        render_rag_answer(result)
                    except Exception as e:
                        st.error(f"Claude RAG error: {e}")
        else:
            render_alert("✅ No erythrocytosis detected.", "g")

    with tabs[5]:
        st.markdown("#### Other Findings")
        found   = False
        eos_v   = data.get("eos_abs")
        wbc_ref = data.get("wbc", 0) or 0
        if eos_v and eos_v > 0.5:
            found = True
            render_alert(f"🟠 **Eosinophilia** — AEC {eos_v:.2f} ×10⁹/L "
                         f"{'(Hypereosinophilia — screen for organ damage)' if eos_v>=1.5 else ''}", "a")
            render_alert("→ Consider: parasites (stool O&P ×3), drug reaction, atopy, IBD, malignancy. Check IgE. ECG if severe.", "b")
        lymph_hi = data.get("lymph_abs") or (wbc_ref * (data.get("lymph_pct") or 0) / 100)
        if lymph_hi and lymph_hi > 4.8:
            found = True
            render_alert(f"🔵 **Lymphocytosis** — ALC {lymph_hi:.2f} ×10⁹/L "
                         f"{'(Suspect CLL if persistent >5)' if lymph_hi>5 else ''}", "a")
            render_alert("→ Reactive (EBV/CMV/viral) vs clonal (CLL: CD5+/CD19+/CD23+). Flow cytometry if >5 ×10⁹/L.", "b")
        if data.get("immature_gran") and data["immature_gran"] > 0:
            found = True
            render_alert(f"⚪ **Immature Granulocytes** {data['immature_gran']:.1f}% — smear; left shift, CML, MDS, sepsis.", "a")
        if data.get("nrbc") and data["nrbc"] > 0:
            found = True
            render_alert(f"🔴 **Nucleated RBCs** {data['nrbc']:.1f}% — marrow infiltration, asplenia, haemolysis, hypoxia, myelofibrosis.", "a")
        if not found:
            render_alert("✅ No other notable findings.", "g")

    # ── 4. COMPREHENSIVE CLAUDE RAG NARRATIVE ─────────────
    if rag_mode == "rag_full" and api_key:
        step_label("4", "Comprehensive Claude RAG Clinical Narrative")
        with st.spinner("◆ Claude is synthesising a full clinical narrative…"):
            try:
                engine = get_rag_engine(api_key)
                result = engine.full_rag_analysis(data, sex, age)
                render_rag_answer(result)
            except Exception as e:
                st.error(f"Comprehensive RAG error: {e}")
                import traceback; st.code(traceback.format_exc())

    # ── 5. KEYWORD RAG (built-in, no API) ─────────────────
    if rag_mode == "built_in" and entered:
        step_label("4", "Knowledge Base Quick Search (Keyword Fallback)")
        kw_parts = []
        if data.get("hgb") and data["hgb"] < (13.5 if sex=="M" else 12.0):
            kw_parts.append("anemia hemoglobin low")
        if data.get("mcv") and data["mcv"] < 80:
            kw_parts.append("microcytic MCV low iron deficiency thalassemia")
        if data.get("mcv") and data["mcv"] > 100:
            kw_parts.append("macrocytic B12 folate MDS")
        if data.get("neut_abs") and data["neut_abs"] > 7.7: kw_parts.append("neutrophilia infection CML")
        if data.get("neut_abs") and data["neut_abs"] < 1.8: kw_parts.append("neutropenia drug autoimmune congenital")
        if data.get("plt") and data["plt"] < 150: kw_parts.append("thrombocytopenia ITP platelet")
        if data.get("lymph_abs") and data["lymph_abs"] < 1.0: kw_parts.append("lymphopenia immunodeficiency SCID")

        if kw_parts:
            retriever = get_keyword_retriever()
            chunks    = retriever.search(" ".join(kw_parts), top_k=3)
            st.markdown('<div style="font-size:.8rem;color:#8b949e;margin-bottom:8px">'
                        'Top 3 relevant knowledge passages (keyword search):</div>', unsafe_allow_html=True)
            for i, chunk in enumerate(chunks, 1):
                pct = int(chunk["_score"] * 100)
                with st.expander(f"📖 [{i}] {chunk['title']} — {chunk['section']} (match: {pct}%)"):
                    st.markdown(f'<div style="font-size:.85rem;color:#e6edf3;line-height:1.7">{chunk["text"]}</div>',
                                unsafe_allow_html=True)

    # ── SUMMARY TABLE ─────────────────────────────────────
    step_label("5", "Abnormal Findings Summary")
    checks = [
        ("Hemoglobin",      data.get("hgb"),      hgb_lo, hgb_hi, "g/dL"),
        ("WBC",             data.get("wbc"),       4.5,    11.0,   "×10⁹/L"),
        ("Platelets",       data.get("plt"),       150,    400,    "×10⁹/L"),
        ("MCV",             data.get("mcv"),       80,     100,    "fL"),
        ("MCHC",            data.get("mchc"),      32,     36,     "g/dL"),
        ("RDW",             data.get("rdw"),       11.5,   14.5,   "%"),
        ("Neutrophils Abs", data.get("neut_abs"),  1.8,    7.7,    "×10⁹/L"),
        ("Lymphocytes Abs", data.get("lymph_abs"), 1.0,    4.8,    "×10⁹/L"),
        ("Eosinophils Abs", data.get("eos_abs"),   0.0,    0.5,    "×10⁹/L"),
        ("MPV",             data.get("mpv"),       7.5,    12.5,   "fL"),
        ("Reticulocytes",   data.get("retic"),     0.5,    2.5,    "%"),
    ]
    rows = []
    for name, val, lo, hi, unit in checks:
        if val is None: continue
        status = classify_value(val, lo, hi)
        if status != "ok":
            rows.append({
                "Parameter": name,
                "Value":     f"{val:.2f} {unit}",
                "Reference": f"{lo}–{hi} {unit}",
                "Status":    "⬇ Low" if "low" in status else "⬆ High",
            })
    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        render_alert("✅ All entered parameters within reference ranges.", "g")

    st.markdown(
        '<div class="footer">🧬 CBC RAG Analyzer · UpToDate Knowledge Base · '
        'all-MiniLM-L6-v2 Local Embeddings · Claude claude-3-5-haiku · For clinical decision support only</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
