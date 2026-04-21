from __future__ import annotations
import os
import io
import time
import random
import math
import asyncio
from typing import List, Dict, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import requests
import httpx

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

# Redaction profile names (patterns defined in _REDACTION_RULES below)
REDACTION_PROFILE_NAMES = ["None", "PII", "PCI", "GDPR", "All (PII + PCI + GDPR)"]

# Session keys
S_ORIGINAL_TEXT = "original_text"
S_REDACTED_TEXT = "redacted_text"
S_REDACTION_LOG = "redaction_log"
S_TRANSLATED_TEXT = "translated_text"
S_PIVOT_KEYWORDS = "pivot_keywords"
S_TRANSLATED_KEYWORDS = "translated_keywords"
S_UPLOAD_META = "upload_meta"

# ===============================================================
# Helpers
# ===============================================================

def ensure_nltk_resources():
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
# PII / PCI / GDPR Redaction — pure regex, zero extra dependencies
# ===============================================================
import re

# Each entry: (entity_type, compiled_pattern, profiles_it_applies_to)
_REDACTION_RULES: List[Tuple[str, re.Pattern, List[str]]] = [
    # PCI
    ("CREDIT_CARD",    re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?'
                                   r'|5[1-5][0-9]{14}'
                                   r'|3[47][0-9]{13}'
                                   r'|6(?:011|5[0-9]{2})[0-9]{12}'
                                   r'|(?:2131|1800|35\d{3})\d{11})\b'),
                                   ["PCI", "All (PII + PCI + GDPR)"]),
    ("IBAN",           re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b'),
                                   ["PCI", "All (PII + PCI + GDPR)"]),
    ("US_SSN",         re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                                   ["PCI", "All (PII + PCI + GDPR)"]),
    ("US_BANK_ACCT",   re.compile(r'\b\d{8,17}\b'),
                                   ["PCI", "All (PII + PCI + GDPR)"]),
    # PII
    ("EMAIL",          re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
                                   ["PII", "GDPR", "All (PII + PCI + GDPR)"]),
    ("PHONE",          re.compile(r'(?<!\d)(?:\+?\d[\s\-.]?)?'
                                   r'(?:\(?\d{3}\)?[\s\-.]?)?\d{3}[\s\-.]?\d{4}(?!\d)'),
                                   ["PII", "All (PII + PCI + GDPR)"]),
    ("IP_ADDRESS",     re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
                                   ["PII", "GDPR", "All (PII + PCI + GDPR)"]),
    ("DATE",           re.compile(r'\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}'
                                   r'|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b'),
                                   ["PII", "GDPR", "All (PII + PCI + GDPR)"]),
    ("PASSPORT",       re.compile(r'\b[A-Z]{1,2}[0-9]{6,9}\b'),
                                   ["PII", "GDPR", "All (PII + PCI + GDPR)"]),
    ("URL",            re.compile(r'https?://[^\s]+'),
                                   ["GDPR", "All (PII + PCI + GDPR)"]),
]


def redact_text(text: str, profile: str) -> Tuple[str, List[Dict]]:
    """
    Apply regex-based redaction for the selected profile.
    Returns (redacted_text, audit_log).
    Audit log = list of {entity_type, matched_value, start, end}.
    Works entirely in-process — no external dependencies.
    """
    if profile == "None":
        return text, []

    audit: List[Dict] = []
    for entity_type, pattern, applicable_profiles in _REDACTION_RULES:
        if profile not in applicable_profiles:
            continue
        for match in pattern.finditer(text):
            audit.append({
                "entity_type": entity_type,
                "matched_value": match.group()[:6] + "***",  # partial for audit safety
                "start": match.start(),
                "end": match.end(),
            })
        text = pattern.sub(f"<{entity_type}>", text)

    return text, audit


def redact_chunks(chunks: List[str], profile: str) -> Tuple[List[str], List[Dict]]:
    """Redact a list of text chunks; aggregate audit logs."""
    if profile == "None":
        return chunks, []
    all_audit: List[Dict] = []
    clean: List[str] = []
    for chunk in chunks:
        rc, log = redact_text(chunk, profile)
        clean.append(rc)
        all_audit.extend(log)
    return clean, all_audit


# ===============================================================
# Chunking utilities
# ===============================================================

def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """Greedy whitespace-aware chunker that keeps chunks <= max_chars."""
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    cur = 0
    while cur < len(text):
        end = min(len(text), cur + max_chars)
        if end < len(text):
            window = text[cur:end]
            brk = window.rfind("\n")
            if brk < 0:
                brk = window.rfind(" ")
            if brk > 200:
                end = cur + brk
        chunks.append(text[cur:end])
        cur = end
    return chunks


def batched(iterable: Iterable, n: int) -> Iterable[List]:
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


# ===============================================================
# Backoff wrapper
# ===============================================================

def translate_with_backoff(fn, chunk: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            return fn(chunk)
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err or "limit" in err:
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Translation failed after retries")


# ===============================================================
# Translation — ThreadPoolExecutor (Streamlit Share safe)
# ===============================================================

def translate_chunks_threaded(
    chunks: List[str],
    source: str,
    target: str,
    backend: str,
    concurrency: int = 5,
    progress_bar=None,
) -> List[str]:
    """
    Translate chunks in parallel using ThreadPoolExecutor.
    Safe on Streamlit Share — no asyncio.run() or nested event loops.
    """
    total = len(chunks)
    results: List[str] = [""] * total
    completed = 0

    if backend.startswith("Google"):
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=source, target=target)

        def _translate_one(idx_chunk):
            idx, chunk = idx_chunk
            translated = translate_with_backoff(translator.translate, chunk)
            return idx, translated

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_translate_one, (i, c)): i for i, c in enumerate(chunks)}
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
                completed += 1
                if progress_bar:
                    progress_bar.progress(
                        int(completed / total * 100),
                        text=f"Translating chunk {completed}/{total}…"
                    )

    else:
        # Azure: batch endpoint, thread per batch
        key = os.getenv("AZURE_TRANSLATOR_KEY")
        region = os.getenv("AZURE_TRANSLATOR_REGION")
        endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
        if not key or not region:
            raise RuntimeError("Azure requires AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_REGION env vars.")

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

        def _azure_batch(batch_idx_chunks):
            idxs, batch_chunks = zip(*batch_idx_chunks)
            for attempt in range(4):
                try:
                    resp = requests.post(
                        url, headers=headers,
                        json=[{"text": t} for t in batch_chunks],
                        timeout=60
                    )
                    resp.raise_for_status()
                    translations = [item["translations"][0]["text"] for item in resp.json()]
                    return list(zip(idxs, translations))
                except Exception as e:
                    if attempt == 3:
                        raise
                    time.sleep(1.5 * (attempt + 1))

        batches = list(batched(list(enumerate(chunks)), 25))
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_azure_batch, b) for b in batches]
            for future in as_completed(futures):
                for idx, text in future.result():
                    results[idx] = text
                completed += len(futures[0].result()) if futures else 0  # approx
                if progress_bar:
                    done = sum(1 for r in results if r)
                    progress_bar.progress(
                        min(int(done / total * 100), 99),
                        text=f"Translating batch… {done}/{total} chunks"
                    )

    if progress_bar:
        progress_bar.progress(100, text="Done ✓")
    return results


def translate_large_text(
    text: str, source: str, target: str, backend: str,
    max_chars: int, batch_size: int, concurrency: int,
    progress, redaction_profile: str = "None"
) -> Tuple[str, List[Dict]]:
    """
    Full pipeline: chunk → (optional redact) → translate.
    Returns (translated_text, audit_log).
    """
    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        return "", []

    bar = progress.progress(0, text="Preparing…")

    # Redact before sending to external API
    if redaction_profile != "None":
        bar.progress(5, text="Redacting sensitive data…")
        chunks, audit_log = redact_chunks(chunks, redaction_profile)
    else:
        audit_log = []

    translated = translate_chunks_threaded(
        chunks=chunks,
        source=source,
        target=target,
        backend=backend,
        concurrency=concurrency,
        progress_bar=bar,
    )
    return "".join(translated), audit_log


def translate_list_threaded(
    items: List[str], source: str, target: str, backend: str, concurrency: int = 5
) -> List[str]:
    """Translate a list of keyword strings in parallel."""
    if not items:
        return []
    return translate_chunks_threaded(items, source, target, backend, concurrency)


# ===============================================================
# Keyword Extraction
# ===============================================================

def extract_keywords_rake(text: str, max_phrases: int = 30) -> List[str]:
    if not NLTK_OK:
        raise RuntimeError("rake-nltk / nltk not installed.")
    ensure_nltk_resources()
    r = Rake()
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()
    return phrases[:max_phrases]


def extract_keywords_yake(text: str, max_phrases: int = 30, language: str = "en") -> List[str]:
    if not YAKE_OK:
        raise RuntimeError("yake not installed.")
    kw_extractor = yake.KeywordExtractor(lan=(language.split("-")[0] or "en"), n=3, top=max_phrases)
    candidates = kw_extractor.extract_keywords(text)
    phrases = [p for p, _ in sorted(candidates, key=lambda x: x[1])]
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
    df = pd.DataFrame([{"text": text}])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.read()

def audit_log_to_csv(log: List[Dict]) -> bytes:
    if not log:
        return pd.DataFrame(columns=["entity_type","start","end","score"]).to_csv(index=False).encode("utf-8")
    return pd.DataFrame(log).to_csv(index=False).encode("utf-8")


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
    st.subheader("Performance controls")
    chunk_size = st.slider("Chunk size (chars)", min_value=300, max_value=4000, value=4000, step=100,
                           help="Larger = fewer API calls. 4000 is optimal for Google free tier.")
    batch_size = st.slider("Batch size (Azure)", min_value=1, max_value=50, value=25, step=1)
    concurrency = st.slider("Concurrency", min_value=1, max_value=16, value=5, step=1,
                            help="5 is the safe ceiling for Google free tier on Streamlit Share.")

    st.markdown("---")
    st.subheader("🔒 Redaction")
    redaction_profile = st.selectbox(
        "Redaction profile",
        REDACTION_PROFILE_NAMES,
        index=0,
        help="Applied BEFORE text is sent to translation API."
    )
    if redaction_profile != "None":
        st.success(f"Active: {redaction_profile} — entities redacted before API call.")

    st.markdown("---")
    kw_method = st.radio("Keyword method", ["RAKE", "YAKE"], index=1)
    kw_count = st.slider("Max keywords", min_value=5, max_value=100, value=30, step=5)


# Initialize session state
for key, default in [
    (S_ORIGINAL_TEXT, ""),
    (S_REDACTED_TEXT, ""),
    (S_REDACTION_LOG, []),
    (S_TRANSLATED_TEXT, ""),
    (S_PIVOT_KEYWORDS, []),
    (S_TRANSLATED_KEYWORDS, {}),
    (S_UPLOAD_META, {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Tabs
upload_tab, translation_tab, keywords_tab, export_tab = st.tabs(
    ["1) Upload & Redact", "2) Translation", "3) Keywords", "4) Export"]
)

# -----------------------------
# 1) Upload & Redact
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

    # Inline redaction preview
    if st.session_state[S_ORIGINAL_TEXT] and redaction_profile != "None":
        if st.button("Preview Redaction"):
            with st.spinner("Redacting…"):
                redacted, log = redact_text(
                    st.session_state[S_ORIGINAL_TEXT][:5000], redaction_profile
                )
            st.text_area("Redacted preview (first 5000 chars)", value=redacted, height=200, disabled=True)
            st.caption(f"{len(log)} entity hit(s) detected in preview sample.")

    # Downloads (Original)
    if st.session_state[S_ORIGINAL_TEXT]:
        st.markdown("#### Download original")
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
    st.subheader("Translate transcript (chunked, threaded, backoff)")

    if not st.session_state[S_ORIGINAL_TEXT]:
        st.warning("Please upload or paste a transcript in the Upload tab.")
    else:
        if redaction_profile != "None":
            st.info(f"🔒 Redaction profile **{redaction_profile}** will be applied before sending to API.")

        if st.button("Translate to Pivot Language"):
            try:
                translated, audit = translate_large_text(
                    text=st.session_state[S_ORIGINAL_TEXT],
                    source=src_lang,
                    target=pivot_lang,
                    backend=backend,
                    max_chars=chunk_size,
                    batch_size=batch_size,
                    concurrency=concurrency,
                    progress=st,
                    redaction_profile=redaction_profile,
                )
                st.session_state[S_TRANSLATED_TEXT] = translated
                st.session_state[S_REDACTION_LOG] = audit
                if audit:
                    st.success(f"Translated ✓ — {len(audit)} sensitive entity/entities redacted before sending.")
                else:
                    st.success(f"Translated to {LANGUAGES.get(pivot_lang, pivot_lang)} ✓")
            except Exception as e:
                st.error(f"Translation failed: {e}")

        if st.session_state[S_TRANSLATED_TEXT]:
            st.text_area("Translation (read-only)", value=st.session_state[S_TRANSLATED_TEXT][:20000], height=250, disabled=True)

            if st.session_state[S_REDACTION_LOG]:
                with st.expander(f"🔒 Redaction audit log ({len(st.session_state[S_REDACTION_LOG])} hits)"):
                    st.dataframe(pd.DataFrame(st.session_state[S_REDACTION_LOG]))
                    st.download_button(
                        "Download audit log CSV",
                        data=audit_log_to_csv(st.session_state[S_REDACTION_LOG]),
                        file_name="redaction_audit.csv"
                    )

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
                        out_words = translate_list_threaded(
                            st.session_state[S_PIVOT_KEYWORDS],
                            pivot_lang, lang, backend,
                            concurrency=max(2, concurrency // 2)
                        )
                        translated_map[lang] = out_words
                    st.session_state[S_TRANSLATED_KEYWORDS] = translated_map
                    st.success("Keywords translated ✓")
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
    audit = st.session_state[S_REDACTION_LOG]

    colA, colB, colC, colD, colE = st.columns(5)
    with colA:
        st.metric("Original length", f"{len(orig)} chars")
    with colB:
        st.metric("Translation length", f"{len(trans)} chars")
    with colC:
        st.metric("Pivot keywords", f"{len(kws)}")
    with colD:
        st.metric("Languages for keywords", f"{len(trans_kws)}")
    with colE:
        st.metric("Redacted entities", f"{len(audit)}")

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

    if audit:
        st.markdown("### Redaction Audit Log")
        st.dataframe(pd.DataFrame(audit))
        st.download_button(
            "Download Redaction Audit CSV",
            data=audit_log_to_csv(audit),
            file_name="redaction_audit.csv"
        )

st.markdown("---")
st.caption(
    "Chunked + threaded translation with exponential backoff. "
    "PII/PCI/GDPR redaction via Microsoft Presidio — applied before text leaves your environment. "
    "Adjust performance controls in the sidebar (chunk=4000, concurrency=5 recommended for Streamlit Share)."
)
