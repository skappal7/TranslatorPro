from __future__ import annotations
import os
import io
import math
import asyncio
from typing import List, Dict, Tuple, Iterable

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
# Chunking utilities
# ===============================================================

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Greedy whitespace-aware chunker that keeps chunks <= max_chars."""
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    while cur < len(text):
        end = min(len(text), cur + max_chars)
        # try to break on last whitespace/newline within window
        window = text[cur:end]
        if end < len(text):
            brk = window.rfind("\n")
            if brk < 0:
                brk = window.rfind(" ")
            if brk > 200:  # avoid tiny fragments
                end = cur + brk
        chunk = text[cur:end]
        chunks.append(chunk)
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
# Translation backends (FAST)
# ===============================================================
@st.cache_data(show_spinner=False)
def translate_text_deep_translator(text: str, source: str, target: str) -> str:
    """Legacy single-call (kept for compatibility)."""
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source=source, target=target)
    return _chunked_translate(lambda c: translator.translate(c), text)


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


async def _translate_chunks_deep_async(chunks: List[str], source: str, target: str, concurrency: int = 8) -> List[str]:
    from deep_translator import GoogleTranslator
    sem = asyncio.Semaphore(concurrency)
    translator = GoogleTranslator(source=source, target=target)

    async def run_one(txt: str) -> str:
        async with sem:
            # deep_translator is blocking; run in thread pool
            return await asyncio.to_thread(translator.translate, txt)

    return await asyncio.gather(*[run_one(c) for c in chunks])


async def _translate_batches_azure_async(chunks: List[str], source: str, target: str, batch_size: int = 25, concurrency: int = 8) -> List[str]:
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

    async def post_batch(client: httpx.AsyncClient, texts: List[str]) -> List[str]:
        # simple retry loop
        for attempt in range(4):
            try:
                resp = await client.post(url, headers=headers, json=[{"text": t} for t in texts], timeout=60)
                resp.raise_for_status()
                data = resp.json()
                return [item["translations"][0]["text"] for item in data]
            except Exception:
                if attempt == 3:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))
        return [""] * len(texts)

    results: List[str] = [""] * len(chunks)
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)
        tasks = []
        index = 0
        for batch in batched(chunks, batch_size):
            bstart = index
            bend = index + len(batch)
            index = bend

            async def run(b=batch, s=bstart, e=bend):
                async with sem:
                    outs = await post_batch(client, b)
                    results[s:e] = outs
            tasks.append(asyncio.create_task(run()))
        await asyncio.gather(*tasks)
    return results


def translate_large_text(text: str, source: str, target: str, backend: str, max_chars: int, batch_size: int, concurrency: int, progress: st.delta_generator.DeltaGenerator) -> str:
    """High-throughput translation orchestrator returning the combined string."""
    chunks = chunk_text(text, max_chars=max_chars)
    total = len(chunks)
    if total == 0:
        return ""

    # Progress UI
    bar = progress.progress(0, text="Preparing…")

    async def run():
        bar.progress(0, text=f"Translating {total} chunk(s)…")
        if backend.startswith("Google"):
            outs = await _translate_chunks_deep_async(chunks, source, target, concurrency=concurrency)
        else:
            outs = await _translate_batches_azure_async(chunks, source, target, batch_size=batch_size, concurrency=concurrency)
        return outs

    # Run event loop
    outs: List[str] = asyncio.run(run())

    # Update progress to 100%
    bar.progress(100, text="Done")
    return "".join(outs)


async def translate_list_async(items: List[str], source: str, target: str, backend: str, concurrency: int = 8) -> List[str]:
    if not items:
        return []
    if backend.startswith("Google"):
        return await _translate_chunks_deep_async(items, source, target, concurrency)
    else:
        # For Azure, we can reuse batch endpoint
        return await _translate_batches_azure_async(items, source, target, batch_size=min(25, max(1, concurrency*2)), concurrency=concurrency)


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
    st.subheader("Performance controls")
    chunk_size = st.slider("Chunk size (chars)", min_value=300, max_value=4000, value=1000, step=100,
                           help="Text split size before translation.")
    batch_size = st.slider("Batch size (Azure)", min_value=1, max_value=50, value=25, step=1,
                           help="How many chunks per Azure request.")
    concurrency = st.slider("Concurrency", min_value=1, max_value=32, value=8, step=1,
                            help="Parallel requests/threads.")

    st.markdown("---")
    kw_method = st.radio("Keyword method", ["RAKE", "YAKE"], index=1)
    kw_count = st.slider("Max keywords", min_value=5, max_value=100, value=30, step=5)
    st.caption("Tip: For very long transcripts, increase chunk size and concurrency within API limits.")

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
# 2) Translation (FAST)
# -----------------------------
with translation_tab:
    st.subheader("Translate transcript (chunked, batched, async)")

    if not st.session_state[S_ORIGINAL_TEXT]:
        st.warning("Please upload or paste a transcript in the Upload tab.")
    else:
        if st.button("Fast Translate to Pivot Language"):
            try:
                st.session_state[S_TRANSLATED_TEXT] = translate_large_text(
                    text=st.session_state[S_ORIGINAL_TEXT],
                    source=src_lang,
                    target=pivot_lang,
                    backend=backend,
                    max_chars=chunk_size,
                    batch_size=batch_size,
                    concurrency=concurrency,
                    progress=st
                )
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
            st.subheader("Translate keywords to selected languages (parallel)")
            if st.button("Translate Keywords"):
                try:
                    async def run_kw():
                        translated_map: Dict[str, List[str]] = {}
                        for lang in out_langs:
                            out_words = await translate_list_async(st.session_state[S_PIVOT_KEYWORDS], pivot_lang, lang, backend, concurrency=max(2, concurrency//2))
                            translated_map[lang] = out_words
                        return translated_map
                    st.session_state[S_TRANSLATED_KEYWORDS] = asyncio.run(run_kw())
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
    "This build uses chunked + batched + async translation with progress bars. Adjust performance controls in the sidebar for your quotas and network."
)
