#app.py — Streamlit Multilingual Transcript Translator & Keyword Pipeline
Python: 3.11-compatible

Features
- Tabs: Upload → Translation → Keywords → Export
- Upload TXT/CSV/XLSX, choose text columns for tabular files
- Source language (or Auto), pivot language for keyword extraction, multiple target languages for keyword back-translation
- Translation backends: deep-translator (Google) or Azure Translator (REST). Azure requires env vars
- Keyword extraction: RAKE (nltk) or YAKE
- Download artifacts at each step in TXT/CSV/XLSX
- Caching and chunking for long texts; clear error messages

Run
  pip install -r requirements.txt
  # (optional) set AZURE_TRANSLATOR_KEY, AZURE_TRANSLATOR_REGION, AZURE_TRANSLATOR_ENDPOINT
  streamlit run app.py
"""
from __future__ import annotations
import os
import io
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import requests

# --- Optional keyword extractors ---
try:
    import nltk
    from rake_nltk import Rake
    NLTK_OK = True
except Exception:
    NLTK_OK = False

try:
    import yake
    YAKE_OK = True
except Exception:
    YAKE_OK = False

# ===============================================================
# Constants
# ===============================================================
APP_TITLE = "Multilingual Transcript Translator & Keyword Pipeline"

LANGUAGES = {
    "auto": "Auto Detect",
    "en": "English",
    "ja": "Japanese",
    "hi": "Hindi",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
    "id": "Indonesian",
    "vi": "Vietnamese",
}

DEFAULT_SOURCE_LANG = "ja"
DEFAULT_PIVOT_LANG = "en"
DEFAULT_KEYWORD_BACK_TRANSLATE = ["ja"]
TRANSLATION_BACKENDS = ["Google (deep-translator)", "Azure Translator"]

# Session keys
S_ORIGINAL_TEXT = "original_text"
S_TRANSLATED_TEXT = "translated_text"
S_PIVOT_KEYWORDS = "pivot_keywords"
S_TRANSLATED_KEYWORDS = "translated_keywords"  # dict(lang -> list[str])
S_UPLOAD_META = "upload_meta"

# ===============================================================
# Helpers
# ===============================================================

def ensure_nltk_resources():
    """Ensure nltk punkt + stopwords are present; download once."""
    if not NLTK_OK:
        return
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def read_uploaded_file(file) -> Tuple[str, Dict]:
    name = file.name.lower()
    meta: Dict = {"filename": file.name}
    if name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
        meta.update({"type": "txt"})
        return text, meta
    elif name.endswith(".csv"):
        df = pd.read_csv(file)
        meta.update({"type": "csv", "columns": df.columns.tolist(), "df": df})
        return "", meta
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file, engine="openpyxl")
        meta.update({"type": "excel", "columns": df.columns.tolist(), "df": df})
        return "", meta
    else:
        raise ValueError("Unsupported file type. Please upload TXT, CSV, or Excel.")


def concat_text_from_df(df: pd.DataFrame, columns: List[str]) -> str:
    if not columns:
        return ""
    rows = df[columns].astype(str).agg(" ".join, axis=1)
    return "\n".join(rows.tolist())


# ===============================================================
# Translation backends
# ===============================================================
@st.cache_data(show_spinner=False)
def translate_text_deep_translator(text: str, source: str, target: str) -> str:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source=source, target=target)
    return _chunked_translate(lambda c: translator.translate(c), text)


@st.cache_data(show_spinner=False)
def translate_text_azure(text: str, source: str, target: str) -> str:
    key = os.getenv("AZURE_TRANSLATOR_KEY")
    region = os.getenv("AZURE_TRANSLATOR_REGION")
    endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
    if not key or not region:
        raise RuntimeError("Azure Translator requires AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_REGION env vars.")

    path = "/translate?api-version=3.0"
    params = f"&to={target}"
    if source != "auto":
        params += f"&from={source}"
    url = endpoint + path + params

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-type": "application/json",
    }

    def api_call(chunk: str) -> str:
        resp = requests.post(url, headers=headers, json=[{"text": chunk}], timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data[0]["translations"][0]["text"]

    return _chunked_translate(api_call, text)


def _chunked_translate(fn, text: str, max_len: int = 4500, overlap: int = 0) -> str:
    if len(text) <= max_len:
        return fn(text)
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_len)
        parts.append(fn(text[start:end]))
        start = end - overlap
        if start < 0:
            start = end
    return "".join(parts)


# ===============================================================
# Keyword Extraction
# ===============================================================

def extract_keywords_rake(text: str, max_phrases: int = 30) -> List[str]:
    if not NLTK_OK:
        raise RuntimeError("rake-nltk / nltk not installed. Add 'rake-nltk' and 'nltk' to requirements.")
    ensure_nltk_resources()
    r = Rake()  # uses stopwords + punkt
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()
    return phrases[:max_phrases]


def extract_keywords_yake(text: str, max_phrases: int = 30, language: str = "en") -> List[str]:
    if not YAKE_OK:
        raise RuntimeError("yake not installed. Add 'yake' to requirements.")
    kw_extractor = yake.KeywordExtractor(lan=(language.split("-")[0] or "en"), n=3, top=max_phrases)
    candidates = kw_extractor.extract_keywords(text)
    phrases = [p for p, _ in sorted(candidates, key=lambda x: x[1])]  # lower score is better
    return phrases[:max_phrases]


# ===============================================================
# Export helpers
# ===============================================================

def to_txt_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def list_to_txt_bytes(lines: List[str]) -> bytes:
    return ("\n".join(lines)).encode("utf-8")


def to_csv_bytes(rows: List[str], header: str = "value") -> bytes:
    df = pd.DataFrame(rows, columns=[header])
    return df.to_csv(index=False).encode("utf-8")


def to_csv_text(text: str) -> bytes:
    df = pd.DataFrame([{"text": text}])
    return df.to_csv(index=False).encode("utf-8")


def to_xlsx_bytes(rows: List[str], header: str = "value") -> bytes:
    df = pd.DataFrame(rows, columns=[header])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.read()


def text_to_xlsx_bytes(text: str) -> bytes:
    df = pd.DataFrame([{ "text": text }])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.read()


# ===============================================================
# UI
# ===============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🗣️", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox(
        "Translation backend",
        TRANSLATION_BACKENDS,
        index=0,
        help="Use Azure for enterprise SLAs (set env vars)."
    )
    src_lang = st.selectbox(
        "Source language",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(DEFAULT_SOURCE_LANG),
        format_func=lambda k: LANGUAGES[k]
    )
    pivot_lang = st.selectbox(
        "Pivot language for keywords",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(DEFAULT_PIVOT_LANG),
        format_func=lambda k: LANGUAGES[k]
    )
    out_langs = st.multiselect(
        "Translate keywords into…",
        list(LANGUAGES.keys()),
        default=DEFAULT_KEYWORD_BACK_TRANSLATE,
        format_func=lambda k: LANGUAGES[k]
    )
    st.markdown("---")
    kw_method = st.radio("Keyword method", ["RAKE", "YAKE"], index=0)
    kw_count = st.slider("Max keywords", min_value=5, max_value=100, value=30, step=5)
    st.caption("Tip: For very long transcripts, consider chunking upstream.")

# Initialize session state
for key, default in [
    (S_ORIGINAL_TEXT, ""),
    (S_TRANSLATED_TEXT, ""),
    (S_PIVOT_KEYWORDS, []),
    (S_TRANSLATED_KEYWORDS, {}),
    (S_UPLOAD_META, {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Tabs
upload_tab, translation_tab, keywords_tab, export_tab = st.tabs(["1) Upload", "2) Translation", "3) Keywords", "4) Export"])

# -----------------------------
# 1) Upload
# -----------------------------
with upload_tab:
    st.subheader("Upload transcript (TXT / CSV / Excel)")
    uploaded = st.file_uploader("Choose a file", type=["txt", "csv", "xlsx", "xls"], accept_multiple_files=False)

    chosen_cols: List[str] = []
    if uploaded is not None:
        try:
            text, meta = read_uploaded_file(uploaded)
            st.session_state[S_UPLOAD_META] = meta
            if meta.get("type") == "txt":
                st.text_area("Preview (read-only)", value=text[:4000], height=200, disabled=True)
                st.session_state[S_ORIGINAL_TEXT] = text
            else:
                df = meta["df"]
                st.dataframe(df.head(50))
                st.info("Select column(s) that contain text to process.")
                chosen_cols = st.multiselect("Columns", meta["columns"], meta["columns"][:1])
                if st.button("Build transcript from selected columns"):
                    built = concat_text_from_df(df, chosen_cols)
                    if not built.strip():
                        st.warning("No text found in the selected columns.")
                    else:
                        st.success(f"Built transcript from {len(chosen_cols)} column(s), {len(built)} characters.")
                        st.text_area("Preview (read-only)", value=built[:4000], height=200, disabled=True)
                        st.session_state[S_ORIGINAL_TEXT] = built
        except Exception as e:
            st.error(f"Upload error: {e}")

    st.markdown("### Or paste text")
    pasted = st.text_area("Paste transcript here (optional)", height=150)
    if pasted:
        st.session_state[S_ORIGINAL_TEXT] = pasted
        st.session_state[S_UPLOAD_META] = {"type": "pasted", "filename": "pasted.txt"}

    # Downloads (Original)
    if st.session_state[S_ORIGINAL_TEXT]:
        st.markdown("#### Download original (from current session)")
        orig = st.session_state[S_ORIGINAL_TEXT]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download TXT", data=to_txt_bytes(orig), file_name="original.txt")
        with col2:
            st.download_button("Download CSV", data=to_csv_text(orig), file_name="original.csv")
        with col3:
            st.download_button("Download XLSX", data=text_to_xlsx_bytes(orig), file_name="original.xlsx")

# -----------------------------
# 2) Translation
# -----------------------------
with translation_tab:
    st.subheader("Translate transcript")

    if not st.session_state[S_ORIGINAL_TEXT]:
        st.warning("Please upload or paste a transcript in the Upload tab.")
    else:
        def do_translate(text: str, from_lang: str, to_lang: str) -> str:
            if backend.startswith("Google"):
                return translate_text_deep_translator(text, from_lang, to_lang)
            else:
                return translate_text_azure(text, from_lang, to_lang)

        if st.button("Translate to Pivot Language"):
            with st.spinner("Translating…"):
                try:
                    translated = do_translate(st.session_state[S_ORIGINAL_TEXT], src_lang, pivot_lang)
                    st.session_state[S_TRANSLATED_TEXT] = translated
                    st.success(f"Translated to {LANGUAGES.get(pivot_lang, pivot_lang)}")
                except Exception as e:
                    st.error(f"Translation failed: {e}")

        if st.session_state[S_TRANSLATED_TEXT]:
            st.text_area("Translation (read-only)", value=st.session_state[S_TRANSLATED_TEXT][:20000], height=250, disabled=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("Download TXT", data=to_txt_bytes(st.session_state[S_TRANSLATED_TEXT]), file_name=f"translation_{pivot_lang}.txt")
            with col2:
                st.download_button("Download CSV", data=to_csv_text(st.session_state[S_TRANSLATED_TEXT]), file_name=f"translation_{pivot_lang}.csv")
            with col3:
                st.download_button("Download XLSX", data=text_to_xlsx_bytes(st.session_state[S_TRANSLATED_TEXT]), file_name=f"translation_{pivot_lang}.xlsx")

# -----------------------------
# 3) Keywords
# -----------------------------
with keywords_tab:
    st.subheader("Extract keywords and translate them")

    if not st.session_state[S_TRANSLATED_TEXT]:
        st.warning("Please generate a translation in the Translation tab.")
    else:
        text_for_kw = st.session_state[S_TRANSLATED_TEXT]

        if st.button("Extract Keywords"):
            try:
                if kw_method == "RAKE":
                    kws = extract_keywords_rake(text_for_kw, kw_count)
                else:
                    kws = extract_keywords_yake(text_for_kw, kw_count, language=pivot_lang)
                # dedupe preserve order
                seen = set()
                uniq = []
                for k in kws:
                    k2 = k.strip()
                    if k2 and k2.lower() not in seen:
                        seen.add(k2.lower())
                        uniq.append(k2)
                st.session_state[S_PIVOT_KEYWORDS] = uniq
                st.success(f"Extracted {len(uniq)} keyword(s).")
            except Exception as e:
                st.error(f"Keyword extraction failed: {e}")

        if st.session_state[S_PIVOT_KEYWORDS]:
            st.write(st.session_state[S_PIVOT_KEYWORDS])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("Download TXT", data=list_to_txt_bytes(st.session_state[S_PIVOT_KEYWORDS]), file_name="keywords_pivot.txt")
            with col2:
                st.download_button("Download CSV", data=to_csv_bytes(st.session_state[S_PIVOT_KEYWORDS], header="keyword"), file_name="keywords_pivot.csv")
            with col3:
                st.download_button("Download XLSX", data=to_xlsx_bytes(st.session_state[S_PIVOT_KEYWORDS], header="keyword"), file_name="keywords_pivot.xlsx")

            st.markdown("---")
            st.subheader("Translate keywords to selected languages")
            if st.button("Translate Keywords"):
                try:
                    translated_map: Dict[str, List[str]] = {}
                    for lang in out_langs:
                        out_words: List[str] = []
                        for kw in st.session_state[S_PIVOT_KEYWORDS]:
                            if backend.startswith("Google"):
                                out_words.append(translate_text_deep_translator(kw, pivot_lang, lang))
                            else:
                                out_words.append(translate_text_azure(kw, pivot_lang, lang))
                        translated_map[lang] = out_words
                    st.session_state[S_TRANSLATED_KEYWORDS] = translated_map
                    st.success("Keywords translated.")
                except Exception as e:
                    st.error(f"Keyword translation failed: {e}")

            if st.session_state[S_TRANSLATED_KEYWORDS]:
                for lang, words in st.session_state[S_TRANSLATED_KEYWORDS].items():
                    with st.expander(f"{LANGUAGES.get(lang, lang)} keywords"):
                        st.write(words)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.download_button("TXT", data=list_to_txt_bytes(words), file_name=f"keywords_{lang}.txt")
                        with c2:
                            st.download_button("CSV", data=to_csv_bytes(words, header="keyword"), file_name=f"keywords_{lang}.csv")
                        with c3:
                            st.download_button("XLSX", data=to_xlsx_bytes(words, header="keyword"), file_name=f"keywords_{lang}.xlsx")

# -----------------------------
# 4) Export
# -----------------------------
with export_tab:
    st.subheader("Summary & Bulk Exports")
    orig = st.session_state[S_ORIGINAL_TEXT]
    trans = st.session_state[S_TRANSLATED_TEXT]
    kws = st.session_state[S_PIVOT_KEYWORDS]
    trans_kws = st.session_state[S_TRANSLATED_KEYWORDS]

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Original length", f"{len(orig)} chars")
    with colB:
        st.metric("Translation length", f"{len(trans)} chars")
    with colC:
        st.metric("Pivot keywords", f"{len(kws)}")
    with colD:
        st.metric("Languages for keywords", f"{len(trans_kws)}")

    st.markdown("### Quick downloads")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Original.txt", data=to_txt_bytes(orig or ""), file_name="original.txt")
        st.download_button("Keywords_pivot.txt", data=list_to_txt_bytes(kws or []), file_name="keywords_pivot.txt")
    with c2:
        st.download_button("Translation.csv", data=to_csv_text(trans or ""), file_name=f"translation_{pivot_lang}.csv")
        st.download_button("Keywords_pivot.csv", data=to_csv_bytes(kws or [], header="keyword"), file_name="keywords_pivot.csv")
    with c3:
        st.download_button("Translation.xlsx", data=text_to_xlsx_bytes(trans or ""), file_name=f"translation_{pivot_lang}.xlsx")
        st.download_button("Keywords_pivot.xlsx", data=to_xlsx_bytes(kws or [], header="keyword"), file_name="keywords_pivot.xlsx")

st.markdown("---")
st.caption(
    "Production tips: Prefer Azure Translator for quota/SLAs; use Streamlit secrets for keys; add retry/backoff "
    "and telemetry (e.g., Azure App Insights) as needed."
)
