import os
import re
import time
import base64
import logging
from io import BytesIO

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="🌿",
    layout="wide"
)

logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- OPTIONAL PDF (SAFE) --------------------
PDF_AVAILABLE = True
try:
    from fpdf import FPDF
except Exception:
    PDF_AVAILABLE = False

# -------------------- PATH HELPERS --------------------
def _first_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

# -------------------- LOAD CALIBRATED MODEL (ONLY) --------------------
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "mental_health_model_v2_calibrated.pkl"),
    os.path.join(BASE_DIR, "calibrated.pkl"),
    os.path.join(BASE_DIR, "mental_health_v2_calibrated.pkl"),
    os.path.join(BASE_DIR, "models", "mental_health_model_v2_calibrated.pkl"),
    os.path.join(BASE_DIR, "app", "mental_health_model_v2_calibrated.pkl"),
]
MODEL_PATH = _first_existing_path(MODEL_CANDIDATES)

if not MODEL_PATH:
    st.error(
        "Calibrated model not found.\n\n"
        "✅ Put `mental_health_model_v2_calibrated.pkl` next to `app.py`.\n\n"
        "Tried:\n" + "\n".join([f"- `{p}`" for p in MODEL_CANDIDATES])
    )
    st.stop()

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model at `{MODEL_PATH}`.\n\nError: {e}")
    st.stop()

# -------------------- OPTIONAL TRAINING CSV (for better UI options) --------------------
TRAINING_DATA_CANDIDATES = [
    os.path.abspath(os.path.join(BASE_DIR, "..", "data", "us_model_data", "training_model_dataset.csv")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "us_model_data", "training_model_dataset.csv")),
    os.path.join(BASE_DIR, "training_model_dataset.csv"),
]
TRAINING_PATH = _first_existing_path(TRAINING_DATA_CANDIDATES)

@st.cache_data(show_spinner=False)
def _load_training_df(path):
    if not path:
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None

TRAINING_DF = _load_training_df(TRAINING_PATH)

# -------------------- INTROSPECT MODEL (columns + numeric/categorical) --------------------
def _find_underlying_estimator(m):
    """
    CalibratedClassifierCV stores fitted estimators in calibrated_classifiers_.
    We use the first one as representative to extract preprocessing info safely.
    """
    # fitted CalibratedClassifierCV
    if hasattr(m, "calibrated_classifiers_") and getattr(m, "calibrated_classifiers_"):
        cc = m.calibrated_classifiers_[0]
        if hasattr(cc, "estimator"):
            return cc.estimator
    # sometimes stored as .estimator
    if hasattr(m, "estimator"):
        return m.estimator
    return m

def _extract_preprocessor_info(m):
    """
    Returns:
      - required_cols: list[str]
      - numeric_cols: set[str]
      - categorical_cols: set[str]
    This is robust across Pipeline + ColumnTransformer setups.
    """
    est = _find_underlying_estimator(m)

    required_cols = []
    numeric_cols = set()
    categorical_cols = set()

    # Best: feature_names_in_ on the estimator/pipeline
    if hasattr(est, "feature_names_in_"):
        required_cols = list(est.feature_names_in_)

    # Try to locate ColumnTransformer inside a pipeline
    pre = None
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        if isinstance(est, Pipeline):
            for _, step in est.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    pre = step
                    break
    except Exception:
        pre = None

    # If we found ColumnTransformer, extract the column lists from its configured transformers
    if pre is not None and hasattr(pre, "transformers"):
        for name, _transformer, cols in pre.transformers:
            # cols can be list[str], np.array, slice, boolean mask, callable -> only accept explicit lists
            if isinstance(cols, (list, tuple, np.ndarray)):
                cols_list = list(cols)
                n = str(name).lower()
                if "num" in n or "numeric" in n:
                    numeric_cols |= set(cols_list)
                elif "cat" in n or "categor" in n:
                    categorical_cols |= set(cols_list)

        # If required_cols missing, use union
        if not required_cols:
            union_cols = list(dict.fromkeys(list(numeric_cols) + list(categorical_cols)))
            required_cols = union_cols

    # If still missing, final fallback
    if not required_cols:
        required_cols = ["Age", "self_employed", "family_history", "remote_work", "tech_company"]

    return required_cols, numeric_cols, categorical_cols

REQUIRED_COLS, NUMERIC_COLS, CATEGORICAL_COLS = _extract_preprocessor_info(MODEL)

# -------------------- UI LABELS (no raw variable names) --------------------
QUESTION_TEXT = {
    "Age": "How old are you?",
    "self_employed": "Are you self-employed?",
    "family_history": "Do you have a family history of mental illness?",
    "remote_work": "Do you usually work remotely?",
    "tech_company": "Do you work in a tech company?",
    "benefits": "Does your employer provide mental health benefits?",
    "care_options": "Do you know the mental health care options available at work?",
    "leave": "How easy is it to take medical leave for mental health reasons?",
    "anonymity": "Is anonymity protected if you seek mental health support at work?",
    "work_interfere": "If you experienced mental health challenges, would it interfere with your work?",
    "no_employees": "Roughly how large is your company?",
    "gender": "What’s your gender?",
    "Gender": "What’s your gender?",
}

def _pretty_question(col: str) -> str:
    if col in QUESTION_TEXT:
        return QUESTION_TEXT[col]

    base = col.replace("_", " ").strip()
    base = re.sub(r"\s+", " ", base).lower()

    # If it looks like a yes/no field, phrase it as a question
    if base.startswith(("is ", "are ", "do ", "does ", "have ", "has ", "can ", "would ")):
        t = base[0].upper() + base[1:]
        return t if t.endswith("?") else t + "?"

    # Generic nice fallback
    t = base.title()
    return f"Choose the option that best matches: {t}"

def _pretty_label(col: str) -> str:
    if col in QUESTION_TEXT:
        return QUESTION_TEXT[col].rstrip("?")
    return re.sub(r"\s+", " ", col.replace("_", " ").strip()).title()

# -------------------- DATASET-DRIVEN OPTIONS --------------------
def _lower(x): return str(x).strip().lower()

UNKNOWN_TOKENS = {"don't know", "dont know", "unknown", "not sure", "unsure", ""}

def _safe_unique_values(series: pd.Series, max_n=12):
    vc = series.dropna().astype(str).value_counts()
    vals = vc.index.tolist()
    # keep most frequent first
    out, seen = [], set()
    for v in vals:
        v = str(v).strip()
        if v not in seen:
            out.append(v)
            seen.add(v)
        if len(out) >= max_n:
            break
    return out

def _infer_options(col: str):
    # If training df exists and column exists and it looks categorical -> pull options from dataset
    if TRAINING_DF is not None and col in TRAINING_DF.columns:
        s = TRAINING_DF[col]
        if not pd.api.types.is_numeric_dtype(s):
            opts = _safe_unique_values(s, max_n=14)

            # Normalize common yes/no variants
            lowset = {_lower(o) for o in opts}
            if lowset.issubset({"yes", "no"}):
                return ["Yes", "No", "Don't know"]
            if lowset.issubset({"yes", "no", "don't know", "dont know", "unknown", "not sure", "unsure"}):
                return ["Yes", "No", "Don't know"]

            # ensure a "Don't know" option exists for user safety
            cleaned = []
            for o in opts:
                oo = o.strip()
                if _lower(oo) in UNKNOWN_TOKENS:
                    continue
                cleaned.append(oo)
            if "Don't know" not in cleaned:
                cleaned.append("Don't know")
            return cleaned

    # Fallback guesses
    c = col.lower()
    if c in {"self_employed", "family_history", "remote_work", "tech_company"}:
        return ["Yes", "No", "Don't know"]
    if c in {"benefits", "care_options", "anonymity"}:
        return ["Yes", "No", "Don't know"]
    if c == "leave":
        return ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"]
    if c == "work_interfere":
        return ["Never", "Rarely", "Sometimes", "Often", "Don't know"]
    if c in {"gender"}:
        return ["Prefer not to say", "Male", "Female", "Other"]
    if c in {"no_employees"}:
        return ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000", "Don't know"]
    return None

def _default_for_col(col: str):
    if col.lower() == "age":
        return 30
    # if training df exists: mode (most frequent)
    if TRAINING_DF is not None and col in TRAINING_DF.columns:
        s = TRAINING_DF[col]
        if pd.api.types.is_numeric_dtype(s):
            try:
                return int(round(float(s.dropna().median())))
            except Exception:
                return 30
        try:
            mode = s.dropna().astype(str).mode()
            if len(mode) > 0:
                return str(mode.iloc[0])
        except Exception:
            pass
    return "Don't know"

# -------------------- INPUT SANITIZATION (FIXES YOUR ERROR) --------------------
def _to_nan_if_unknown(v):
    if v is None:
        return np.nan
    s = _lower(v)
    if s in UNKNOWN_TOKENS:
        return np.nan
    return v

def _normalize_value(col: str, v):
    """
    Critical: never feed "Don't know" into numeric columns.
    Also: if a yes/no field is numeric in the model, map Yes/No to 1/0.
    """
    v = _to_nan_if_unknown(v)

    if col in NUMERIC_COLS:
        # Handle common yes/no numeric fields
        if isinstance(v, str):
            s = _lower(v)
            if s in {"yes", "y", "true"}:
                return 1.0
            if s in {"no", "n", "false"}:
                return 0.0
        # numeric conversion
        return pd.to_numeric(v, errors="coerce")

    # categorical
    if isinstance(v, float) and np.isnan(v):
        return np.nan
    return str(v).strip()

def _build_input_df(user_answers: dict):
    row = {c: np.nan for c in REQUIRED_COLS}
    for c in REQUIRED_COLS:
        # if user didn't answer -> NaN (let pipeline impute)
        if c in user_answers:
            row[c] = _normalize_value(c, user_answers[c])
        else:
            # also do NOT insert strings as defaults; keep NaN for safety
            row[c] = np.nan

    df = pd.DataFrame([row])

    # Ensure numeric cols are numeric dtype (very important)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[REQUIRED_COLS]

def predict_with_threshold(input_df: pd.DataFrame, threshold=0.4):
    probs = MODEL.predict_proba(input_df)[0]
    p1 = float(probs[1])
    pred = 1 if p1 >= threshold else 0
    conf = p1 if pred == 1 else (1.0 - p1)
    ci = 0.05
    return pred, conf, max(conf - ci, 0), min(conf + ci, 1)

# -------------------- REPORT (CRASH-PROOF) --------------------
def _pdf_sanitize(text: str, hard_break_every=30):
    if text is None:
        return ""
    s = str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(rf"(\S{{{hard_break_every}}})", r"\1 ", s)
    s = s.encode("latin-1", "ignore").decode("latin-1", "ignore")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def create_pdf_report(title, answers: dict, prediction_label: str, confidence: float, notes: str):
    if not PDF_AVAILABLE:
        return None
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=14)
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, txt=_pdf_sanitize(title), ln=1, align="C")
        pdf.ln(2)

        pdf.set_font("Arial", size=11)
        for q, a in answers.items():
            pdf.multi_cell(0, 7, _pdf_sanitize(f"{q}: {a}"))

        pdf.ln(1)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, _pdf_sanitize(f"Prediction: {prediction_label}"))
        pdf.multi_cell(0, 8, _pdf_sanitize(f"Confidence (calibrated): {confidence:.2%}"))

        pdf.ln(2)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, _pdf_sanitize(notes))

        out = pdf.output(dest="S")
        pdf_bytes = out.encode("latin-1", errors="ignore") if isinstance(out, str) else out
        return BytesIO(pdf_bytes)
    except Exception:
        return None

def create_text_report(title, answers: dict, prediction_label: str, confidence: float, notes: str):
    lines = [title, "=" * len(title), "", "Answers:"]
    for k, v in answers.items():
        lines.append(f"- {k}: {v}")
    lines += ["", f"Prediction: {prediction_label}", f"Confidence (calibrated): {confidence:.2%}", "", notes]
    return "\n".join(lines)

# -------------------- PREMIUM LIGHT UI + RADIO-STYLES --------------------
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display: none !important;}
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 600px at 10% 15%, rgba(187, 247, 208, 0.65), transparent 60%),
    radial-gradient(900px 600px at 90% 10%, rgba(199, 210, 254, 0.70), transparent 60%),
    radial-gradient(900px 600px at 20% 90%, rgba(254, 215, 170, 0.65), transparent 60%),
    radial-gradient(900px 600px at 85% 85%, rgba(253, 164, 175, 0.55), transparent 60%),
    linear-gradient(180deg, #fbfdff 0%, #f7fbff 35%, #fbfbff 100%) !important;
}

h1, h2, h3, h4 { color: #0b1220; }
p, li, div { color: rgba(11, 18, 32, 0.86); }

.glass {
  background: rgba(255,255,255,0.64);
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 22px;
  padding: 18px 18px 14px 18px;
  box-shadow: 0 16px 40px rgba(2, 6, 23, 0.08), 0 1px 0 rgba(255,255,255,0.55) inset;
  backdrop-filter: blur(10px);
}

.hero-title{
  font-size: 42px;
  font-weight: 900;
  letter-spacing: -0.02em;
  margin-bottom: 6px;
}
.hero-sub{
  font-size: 16px;
  line-height: 1.5;
  color: rgba(11,18,32,0.72);
}

.badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(15,23,42,0.10);
  font-weight: 800;
  color: rgba(11,18,32,0.80);
}

/* Buttons */
div.stButton > button {
  border-radius: 999px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  padding: 0.70rem 1.05rem !important;
  background: rgba(255,255,255,0.78) !important;
  color: rgba(11,18,32,0.92) !important;
  font-weight: 800 !important;
  box-shadow: 0 10px 20px rgba(2,6,23,0.08);
  transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
}
div.stButton > button:hover {
  transform: translateY(-1px);
  background: rgba(255,255,255,0.95) !important;
  box-shadow: 0 14px 26px rgba(2,6,23,0.12);
}
div.stButton > button:active { transform: translateY(0px) scale(0.99); }

/* Sidebar */
[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.66) !important;
  backdrop-filter: blur(12px);
  border-right: 1px solid rgba(15,23,42,0.08);
}

/* Question Canvas */
.qwrap{ display:flex; justify-content:center; }
.qcanvas{ width: min(880px, 94vw); margin-top: 8px; position: relative; }
.qinner{
  position: relative;
  padding: 18px 18px 12px 18px;
  border-radius: 22px;
  background: rgba(255,255,255,0.76);
  border: 1px solid rgba(15,23,42,0.10);
  box-shadow: 0 18px 50px rgba(2,6,23,0.10);
  backdrop-filter: blur(12px);
}

@keyframes stepIn {
  from { opacity: 0; transform: translateY(16px) scale(0.985); filter: blur(2px); }
  to   { opacity: 1; transform: translateY(0px) scale(1); filter: blur(0); }
}
.stepAnim { animation: stepIn .46s cubic-bezier(.2,.8,.2,1) both; }

.qnum{
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(11,18,32,0.52);
}
.qtext{
  font-size: 22px;
  font-weight: 950;
  letter-spacing: -0.01em;
  margin: 6px 0 6px 0;
  color: rgba(11,18,32,0.95);
}
.qdesc{
  font-size: 13px;
  color: rgba(11,18,32,0.62);
  margin-top: 0px;
  margin-bottom: 10px;
}

/* Radio -> segmented pills vibe */
div[role="radiogroup"] > label {
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(15,23,42,0.14);
  border-radius: 999px;
  padding: 10px 14px;
  margin: 6px 8px 6px 0 !important;
  display: inline-flex !important;
  align-items: center;
  gap: 10px;
  box-shadow: 0 10px 20px rgba(2,6,23,0.07);
}
div[role="radiogroup"] > label:hover {
  background: rgba(255,255,255,0.95);
}
div[role="radiogroup"] svg { transform: scale(1.15); }
div[role="radiogroup"] input:checked + div {
  font-weight: 900 !important;
}

/* Result */
.result-box {
  padding: 18px;
  border-left: 6px solid #22c55e;
  background: rgba(255,255,255,0.78);
  margin-top: 18px;
  border-radius: 18px;
  font-size: 18px;
  color: rgba(11,18,32,0.92);
  box-shadow: 0 16px 40px rgba(2,6,23,0.10);
}
</style>
""", unsafe_allow_html=True)

# -------------------- FLOATING SIDEBAR TOGGLE (re-open after closing) --------------------
components.html(
    """
    <div style="position:fixed; left:14px; top:14px; z-index:999999;">
      <button id="kukiSidebarToggle"
        style="
          width:44px;height:44px;border-radius:14px;border:1px solid rgba(15,23,42,0.14);
          background:rgba(255,255,255,0.85);box-shadow:0 12px 24px rgba(2,6,23,0.12);
          cursor:pointer;font-weight:900;font-size:18px;color:rgba(11,18,32,0.9);
        ">☰</button>
    </div>
    <script>
      const clickIf = (selList) => {
        const doc = window.parent.document;
        for (const sel of selList){
          const el = doc.querySelector(sel);
          if (el) { el.click(); return true; }
        }
        // fallback: search by aria-label contains "sidebar"
        const btns = Array.from(doc.querySelectorAll("button"));
        for (const b of btns){
          const a = (b.getAttribute("aria-label")||"").toLowerCase();
          if (a.includes("sidebar")) { b.click(); return true; }
        }
        return false;
      };

      document.getElementById("kukiSidebarToggle").addEventListener("click", () => {
        clickIf([
          // common Streamlit selectors (varies by version)
          'button[data-testid="stSidebarCollapsedControl"]',
          '[data-testid="stSidebarCollapsedControl"] button',
          'button[aria-label="Open sidebar"]',
          'button[aria-label="Close sidebar"]'
        ]);
      });
    </script>
    """,
    height=0
)

# -------------------- SESSION STATE --------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "wiz_idx" not in st.session_state:
    st.session_state.wiz_idx = 0
if "wiz_answers" not in st.session_state:
    st.session_state.wiz_answers = {}
if "result" not in st.session_state:
    st.session_state.result = None

# -------------------- SIDEBAR MENU --------------------
st.sidebar.markdown("### Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = "welcome"
    st.rerun()
if st.sidebar.button("Privacy Policy"):
    st.session_state.page = "privacy"
    st.rerun()
if st.sidebar.button("Documentation"):
    st.session_state.page = "documentation"
    st.rerun()

# -------------------- QUESTION ORDER (dataset-driven: model columns) --------------------
preferred_order = [
    "Age", "gender", "Gender",
    "self_employed", "family_history", "remote_work", "tech_company",
    "no_employees", "benefits", "care_options", "anonymity", "leave", "work_interfere",
]
questions = []
seen = set()
for c in preferred_order:
    if c in REQUIRED_COLS and c not in seen:
        questions.append(c); seen.add(c)
for c in REQUIRED_COLS:
    if c not in seen:
        questions.append(c); seen.add(c)

TOTAL = len(questions)

# -------------------- PAGES --------------------
if st.session_state.page == "welcome":
    st.markdown('<div class="qwrap"><div class="glass" style="width:min(980px, 96vw);">', unsafe_allow_html=True)

    st.markdown(f"<div class='hero-title'>Mental Health Treatment Predictor</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>Your predictions are generated using your <b>calibrated v2 model</b> "
        "and the app asks questions based on the <b>actual model input columns</b>.</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>Loaded model: {os.path.basename(MODEL_PATH)}</span>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    consent = st.toggle("I agree to the Privacy Terms", value=False)
    demo_ack = st.toggle("I understand this is not medical advice", value=False)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Start"):
            if not (consent and demo_ack):
                st.warning("Please enable both toggles to continue.")
            else:
                st.session_state.page = "model"
                st.session_state.wiz_idx = 0
                st.session_state.wiz_answers = {}
                st.session_state.result = None
                st.balloons()
                time.sleep(0.45)
                st.rerun()
    with c2:
        st.markdown(
            "<div class='hero-sub'>Tip: if the training CSV exists, option lists "
            "auto-match your dataset (less guessing, cleaner UI).</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div></div>", unsafe_allow_html=True)

elif st.session_state.page == "privacy":
    st.markdown('<div class="qwrap"><div class="glass" style="width:min(980px, 96vw);">', unsafe_allow_html=True)
    st.markdown("<div class='hero-title' style='font-size:34px;'>Privacy Policy</div>", unsafe_allow_html=True)
    st.markdown("""
    - No personal data is stored by this app.
    - Answers are used only to compute a prediction during your session.
    - This is an educational demo and **not** medical advice or diagnosis.
    """)
    st.markdown("</div></div>", unsafe_allow_html=True)

elif st.session_state.page == "documentation":
    st.markdown('<div class="qwrap"><div class="glass" style="width:min(980px, 96vw);">', unsafe_allow_html=True)
    st.markdown("<div class='hero-title' style='font-size:34px;'>Project Documentation</div>", unsafe_allow_html=True)

    with st.expander("What this app uses", expanded=True):
        st.write("✅ **Calibrated model only** (trustworthy probabilities).")
        st.write("✅ Questions are generated from the **model’s required input columns**.")
        st.write("✅ If available, the training CSV is used to populate realistic answer options.")

    with st.expander("Setup"):
        st.code(
            "pip install streamlit pandas scikit-learn joblib\n"
            "pip install fpdf2   # optional for PDF download\n\n"
            "streamlit run app.py"
        )

    with st.expander("Files"):
        st.markdown(f"- Model: `{os.path.basename(MODEL_PATH)}` (must be next to `app.py`)")
        if TRAINING_PATH:
            st.markdown(f"- Training CSV detected: `{TRAINING_PATH}`")
        else:
            st.markdown("- Training CSV not detected (UI will use safe fallback options).")

    with st.expander("Why calibrated matters"):
        st.write(
            "Calibrated models produce better probability estimates than raw scores, which is important "
            "when you display ‘confidence’ to users."
        )

    st.markdown("</div></div>", unsafe_allow_html=True)

elif st.session_state.page == "model":
    idx = int(st.session_state.wiz_idx)

    # progress header
    st.markdown('<div class="qwrap">', unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>Question {min(idx+1, TOTAL)} / {TOTAL}</span>", unsafe_allow_html=True)
    st.progress((idx / TOTAL) if TOTAL > 0 else 0.0)

    # REVIEW STEP
    if idx >= TOTAL:
        st.markdown('<div class="qcanvas"><div class="qinner stepAnim">', unsafe_allow_html=True)
        st.markdown("<div class='qnum'>Review</div>", unsafe_allow_html=True)
        st.markdown("<div class='qtext'>Check your answers before predicting</div>", unsafe_allow_html=True)
        st.markdown("<div class='qdesc'>You can go back and edit anything.</div>", unsafe_allow_html=True)

        answers = st.session_state.wiz_answers.copy()
        rows = [{"Question": _pretty_label(c), "Answer": answers.get(c, "—")} for c in questions if c in answers]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No answers yet. Go back and answer at least one question.")

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("⬅ Back"):
                st.session_state.wiz_idx = max(TOTAL - 1, 0)
                st.rerun()
        with c2:
            if st.button("Reset"):
                st.session_state.wiz_idx = 0
                st.session_state.wiz_answers = {}
                st.rerun()
        with c3:
            ready = st.toggle("I confirm these answers are correct", value=False)
            if st.button("Predict Now ✅"):
                if not ready:
                    st.warning("Enable the confirmation toggle to proceed.")
                else:
                    with st.spinner("Making prediction..."):
                        time.sleep(0.6)
                        input_df = _build_input_df(st.session_state.wiz_answers)
                        pred, conf, lo, hi = predict_with_threshold(input_df, threshold=0.4)

                    pred_label = "Likely to Seek Treatment" if pred == 1 else "Unlikely to Seek Treatment"
                    out_answers = { _pretty_label(k): v for k, v in st.session_state.wiz_answers.items() }

                    st.session_state.result = {
                        "prediction": int(pred),
                        "confidence": float(conf),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "answers": out_answers
                    }
                    st.session_state.page = "results"
                    st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)  # close qinner/qcanvas
        st.markdown("</div>", unsafe_allow_html=True)       # close qwrap

    else:
        col = questions[idx]
        q = _pretty_question(col)

        # canvas card
        st.markdown(f'<div class="qcanvas"><div class="qinner stepAnim" data-step="{idx}">', unsafe_allow_html=True)
        st.markdown(f"<div class='qnum'>Step {idx+1}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='qtext'>{q}</div>", unsafe_allow_html=True)

        default = st.session_state.wiz_answers.get(col, _default_for_col(col))
        opts = _infer_options(col)

        # input control
        answered = False
        if col in NUMERIC_COLS or col.lower() == "age":
            # numeric UI
            if col.lower() == "age":
                v = int(default) if str(default).strip().isdigit() else 30
                v = st.slider(" ", 18, 100, v, label_visibility="collapsed", key=f"age_{idx}")
                st.session_state.wiz_answers[col] = int(v)
                answered = True
            else:
                v0 = default
                try:
                    v0 = float(default)
                except Exception:
                    v0 = 0.0
                v = st.number_input(" ", value=float(v0), label_visibility="collapsed", key=f"num_{col}_{idx}")
                st.session_state.wiz_answers[col] = float(v)
                answered = True

        elif opts is not None:
            # clean segmented radio
            # use horizontal when few options
            horizontal = True if len(opts) <= 4 else False
            v = st.radio(
                " ",
                opts,
                index=opts.index(default) if default in opts else len(opts)-1,
                horizontal=horizontal,
                label_visibility="collapsed",
                key=f"radio_{col}_{idx}"
            )
            st.session_state.wiz_answers[col] = v
            answered = True

        else:
            # free text fallback
            v = st.text_input(" ", value=str(default) if default is not None else "", label_visibility="collapsed", key=f"txt_{col}_{idx}")
            if v.strip():
                st.session_state.wiz_answers[col] = v.strip()
                answered = True

        # nav
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        n1, n2, n3 = st.columns([1, 1, 2])
        with n1:
            if st.button("⬅ Back", key=f"back_{idx}"):
                st.session_state.wiz_idx = max(idx - 1, 0)
                st.rerun()
        with n2:
            if st.button("Skip", key=f"skip_{idx}"):
                st.session_state.wiz_idx = idx + 1
                st.rerun()
        with n3:
            if st.button("Next ➜", key=f"next_{idx}", disabled=not answered):
                st.session_state.wiz_idx = idx + 1
                st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)  # close qinner/qcanvas
        st.markdown("</div>", unsafe_allow_html=True)       # close qwrap

elif st.session_state.page == "results":
    r = st.session_state.result
    if not r:
        st.session_state.page = "model"
        st.rerun()

    prediction = int(r["prediction"])
    confidence = float(r["confidence"])
    ci_low = float(r["ci_low"])
    ci_high = float(r["ci_high"])
    answers = r.get("answers", {})

    st.markdown('<div class="qwrap"><div class="glass" style="width:min(980px, 96vw);">', unsafe_allow_html=True)
    st.markdown("<div class='hero-title' style='font-size:34px;'>Your Prediction Result</div>", unsafe_allow_html=True)

    result_text = "You seem likely to seek mental health support." if prediction == 1 else "You seem unlikely to seek mental health support."
    pred_label = "Likely to Seek Treatment" if prediction == 1 else "Unlikely to Seek Treatment"
    border_color = "#22c55e" if prediction == 1 else "#f59e0b"

    st.markdown(f"""
    <div class="result-box" style="border-left-color:{border_color};">
        <b>Prediction:</b> {result_text}<br>
        <b>Confidence (calibrated):</b> {confidence:.2%} <br>
        <b>Approx. 95% CI:</b> {ci_low:.2%} - {ci_high:.2%}
    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence)
    st.markdown("<div class='hero-sub'>Confidence is based on your <b>calibrated</b> model.</div>", unsafe_allow_html=True)

    with st.expander("Review your answers", expanded=False):
        if answers:
            df_ans = pd.DataFrame([{"Question": k, "Answer": v} for k, v in answers.items()])
            st.dataframe(df_ans, use_container_width=True, hide_index=True)
        else:
            st.info("No answers recorded.")

    title = "Mental Health Prediction Report (Calibrated v2 Model)"
    note = "Educational demo only. Not medical advice, not a diagnosis, and not a substitute for professional support."

    pdf_buf = create_pdf_report(title, answers, pred_label, confidence, note)
    if pdf_buf is not None:
        b64 = base64.b64encode(pdf_buf.getvalue()).decode()
        st.markdown(
            f'<a href="data:application/pdf;base64,{b64}" download="mental_health_report.pdf">'
            f'Download Your Report (PDF)</a>',
            unsafe_allow_html=True
        )
    else:
        txt = create_text_report(title, answers, pred_label, confidence, note)
        b64 = base64.b64encode(txt.encode("utf-8")).decode()
        st.markdown(
            f'<a href="data:text/plain;base64,{b64}" download="mental_health_report.txt">'
            f'Download Your Report (TXT)</a>',
            unsafe_allow_html=True
        )
        if not PDF_AVAILABLE:
            st.info("PDF export is optional. Install with: `pip install fpdf2`")
        else:
            st.info("PDF export was skipped due to font/format limits. TXT is always safe.")

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Try Again"):
            st.session_state.page = "model"
            st.session_state.wiz_idx = 0
            st.session_state.wiz_answers = {}
            st.session_state.result = None
            st.rerun()
    with c2:
        st.markdown("<div class='hero-sub'>Want it even smoother? Next we can add a ‘swipe’ feel + animated progress header.</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    st.session_state.page = "welcome"
    st.rerun()
