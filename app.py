import streamlit as st
import pandas as pd
import numpy as np
from googletrans import Translator
import io
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openpyxl
from datetime import datetime
import base64

# Download NLTK data (run this once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TranscriptProcessor:
    def __init__(self):
        self.translator = Translator()
        self.supported_languages = {
            'Japanese': 'ja',
            'English': 'en',
            'Korean': 'ko',
            'Chinese (Simplified)': 'zh',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Arabic': 'ar',
            'Hindi': 'hi',
            'Thai': 'th',
            'Vietnamese': 'vi'
        }
        
    def detect_language(self, text):
        """Detect the language of the input text"""
        try:
            detection = self.translator.detect(text)
            return detection.lang
        except Exception as e:
            st.error(f"Language detection failed: {str(e)}")
            return None
    
    def translate_text(self, text, src_lang, dest_lang='en'):
        """Translate text from source language to destination language"""
        try:
            if src_lang == dest_lang:
                return text
            result = self.translator.translate(text, src=src_lang, dest=dest_lang)
            return result.text
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
            return text
    
    def extract_keywords(self, text, num_keywords=20):
        """Extract keywords from English text"""
        try:
            # Get English stopwords
            stop_words = set(stopwords.words('english'))
            
            # Tokenize and clean text
            tokens = word_tokenize(text.lower())
            
            # Filter out non-alphabetic tokens and stopwords
            keywords = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
            
            # Count frequency and get top keywords
            keyword_freq = Counter(keywords)
            top_keywords = [word for word, _ in keyword_freq.most_common(num_keywords)]
            
            return top_keywords, keyword_freq
        except Exception as e:
            st.error(f"Keyword extraction failed: {str(e)}")
            return [], {}

def load_file(uploaded_file):
    """Load and parse uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            content = str(uploaded_file.read(), "utf-8")
            df = pd.DataFrame({'text': [content]})
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or TXT files.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_download_link(df, filename, file_format):
    """Create a download link for the dataframe"""
    if file_format == 'CSV':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}.csv</a>'
    elif file_format == 'Excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download {filename}.xlsx</a>'
    elif file_format == 'Text':
        if len(df.columns) == 1:
            text = '\n'.join(df.iloc[:, 0].astype(str))
        else:
            text = df.to_string(index=False)
        b64 = base64.b64encode(text.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="{filename}.txt">Download {filename}.txt</a>'
    
    return href

def main():
    st.set_page_config(
        page_title="Multi-Language Transcript Processor",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Multi-Language Transcript Processor")
    st.markdown("### Professional-grade transcript translation and keyword extraction tool")
    
    # Initialize processor
    processor = TranscriptProcessor()
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    workflow_type = st.sidebar.selectbox(
        "Workflow Type",
        ["Multi-Step Tabs", "Single Page Workflow"]
    )
    
    # Language selection
    source_language = st.sidebar.selectbox(
        "Source Language",
        list(processor.supported_languages.keys()),
        index=0
    )
    
    target_language_back = st.sidebar.selectbox(
        "Target Language for Keywords (back-translation)",
        list(processor.supported_languages.keys()),
        index=0
    )
    
    num_keywords = st.sidebar.slider("Number of Keywords to Extract", 5, 50, 20)
    
    # Initialize session state
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'translated_data' not in st.session_state:
        st.session_state.translated_data = None
    if 'keywords_data' not in st.session_state:
        st.session_state.keywords_data = None
    if 'translated_keywords_data' not in st.session_state:
        st.session_state.translated_keywords_data = None
    
    if workflow_type == "Multi-Step Tabs":
        # Multi-step tab workflow
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "🔄 Translation", "🔑 Keywords", "📤 Export"])
        
        with tab1:
            st.header("Step 1: Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls', 'txt'],
                help="Upload CSV, Excel, or Text files containing transcripts"
            )
            
            if uploaded_file:
                df = load_file(uploaded_file)
                if df is not None:
                    st.session_state.original_data = df
                    st.success("File uploaded successfully!")
                    st.dataframe(df.head())
                    
                    # Download original data
                    st.subheader("Download Original Data")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Download as CSV"):
                            st.markdown(create_download_link(df, "original_data", "CSV"), unsafe_allow_html=True)
                    with col2:
                        if st.button("Download as Excel"):
                            st.markdown(create_download_link(df, "original_data", "Excel"), unsafe_allow_html=True)
                    with col3:
                        if st.button("Download as Text"):
                            st.markdown(create_download_link(df, "original_data", "Text"), unsafe_allow_html=True)
        
        with tab2:
            st.header("Step 2: Translation")
            if st.session_state.original_data is not None:
                df = st.session_state.original_data.copy()
                
                # Select text column
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    text_column = st.selectbox("Select text column to translate", text_columns)
                    
                    if st.button("🔄 Start Translation", type="primary"):
                        progress_bar = st.progress(0)
                        translated_texts = []
                        
                        src_lang_code = processor.supported_languages[source_language]
                        
                        for i, text in enumerate(df[text_column]):
                            if pd.notna(text):
                                translated_text = processor.translate_text(str(text), src_lang_code, 'en')
                                translated_texts.append(translated_text)
                            else:
                                translated_texts.append("")
                            
                            progress_bar.progress((i + 1) / len(df))
                        
                        df['translated_text'] = translated_texts
                        st.session_state.translated_data = df
                        st.success("Translation completed!")
                
                if st.session_state.translated_data is not None:
                    st.subheader("Translation Results")
                    st.dataframe(st.session_state.translated_data)
                    
                    # Download translated data
                    st.subheader("Download Translated Data")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Download CSV (Translation)", key="trans_csv"):
                            st.markdown(create_download_link(st.session_state.translated_data, "translated_data", "CSV"), unsafe_allow_html=True)
                    with col2:
                        if st.button("Download Excel (Translation)", key="trans_excel"):
                            st.markdown(create_download_link(st.session_state.translated_data, "translated_data", "Excel"), unsafe_allow_html=True)
                    with col3:
                        if st.button("Download Text (Translation)", key="trans_text"):
                            st.markdown(create_download_link(st.session_state.translated_data, "translated_data", "Text"), unsafe_allow_html=True)
            else:
                st.warning("Please upload data in the Upload tab first.")
        
        with tab3:
            st.header("Step 3: Keyword Extraction")
            if st.session_state.translated_data is not None:
                if st.button("🔑 Extract Keywords", type="primary"):
                    translated_df = st.session_state.translated_data
                    all_text = ' '.join(translated_df['translated_text'].dropna().astype(str))
                    
                    keywords, keyword_freq = processor.extract_keywords(all_text, num_keywords)
                    
                    # Create keywords dataframe
                    keywords_df = pd.DataFrame({
                        'keyword': keywords,
                        'frequency': [keyword_freq[word] for word in keywords]
                    })
                    
                    # Translate keywords back to target language
                    target_lang_code = processor.supported_languages[target_language_back]
                    translated_keywords = []
                    
                    progress_bar = st.progress(0)
                    for i, keyword in enumerate(keywords):
                        translated_keyword = processor.translate_text(keyword, 'en', target_lang_code)
                        translated_keywords.append(translated_keyword)
                        progress_bar.progress((i + 1) / len(keywords))
                    
                    keywords_df['translated_keyword'] = translated_keywords
                    st.session_state.keywords_data = keywords_df
                    st.session_state.translated_keywords_data = keywords_df
                    
                    st.success("Keywords extracted and translated!")
                
                if st.session_state.keywords_data is not None:
                    st.subheader("Keywords Results")
                    st.dataframe(st.session_state.keywords_data)
                    
                    # Download keywords data
                    st.subheader("Download Keywords Data")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Download CSV (Keywords)", key="kw_csv"):
                            st.markdown(create_download_link(st.session_state.keywords_data, "keywords_data", "CSV"), unsafe_allow_html=True)
                    with col2:
                        if st.button("Download Excel (Keywords)", key="kw_excel"):
                            st.markdown(create_download_link(st.session_state.keywords_data, "keywords_data", "Excel"), unsafe_allow_html=True)
                    with col3:
                        if st.button("Download Text (Keywords)", key="kw_text"):
                            st.markdown(create_download_link(st.session_state.keywords_data, "keywords_data", "Text"), unsafe_allow_html=True)
            else:
                st.warning("Please complete translation in the Translation tab first.")
        
        with tab4:
            st.header("Step 4: Export All Results")
            if st.session_state.translated_keywords_data is not None:
                st.success("All processing steps completed! You can download all results below.")
                
                # Create comprehensive export
                export_data = {
                    'Original Data': st.session_state.original_data,
                    'Translated Data': st.session_state.translated_data,
                    'Keywords Data': st.session_state.keywords_data
                }
                
                for name, data in export_data.items():
                    if data is not None:
                        st.subheader(f"{name}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"CSV", key=f"export_csv_{name}"):
                                st.markdown(create_download_link(data, name.lower().replace(' ', '_'), "CSV"), unsafe_allow_html=True)
                        with col2:
                            if st.button(f"Excel", key=f"export_excel_{name}"):
                                st.markdown(create_download_link(data, name.lower().replace(' ', '_'), "Excel"), unsafe_allow_html=True)
                        with col3:
                            if st.button(f"Text", key=f"export_text_{name}"):
                                st.markdown(create_download_link(data, name.lower().replace(' ', '_'), "Text"), unsafe_allow_html=True)
            else:
                st.warning("Please complete all previous steps to export results.")
    
    else:
        # Single page workflow
        st.header("Single Page Workflow")
        
        # File upload
        st.subheader("📁 Step 1: Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'txt'],
            help="Upload CSV, Excel, or Text files containing transcripts"
        )
        
        if uploaded_file:
            df = load_file(uploaded_file)
            if df is not None:
                st.success("File uploaded successfully!")
                st.dataframe(df.head())
                
                # Original data download
                with st.expander("Download Original Data"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("CSV (Original)", key="single_orig_csv"):
                            st.markdown(create_download_link(df, "original_data", "CSV"), unsafe_allow_html=True)
                    with col2:
                        if st.button("Excel (Original)", key="single_orig_excel"):
                            st.markdown(create_download_link(df, "original_data", "Excel"), unsafe_allow_html=True)
                    with col3:
                        if st.button("Text (Original)", key="single_orig_text"):
                            st.markdown(create_download_link(df, "original_data", "Text"), unsafe_allow_html=True)
                
                # Translation
                st.subheader("🔄 Step 2: Translation")
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    text_column = st.selectbox("Select text column to translate", text_columns)
                    
                    if st.button("🔄 Translate Text", type="primary"):
                        progress_bar = st.progress(0)
                        translated_texts = []
                        src_lang_code = processor.supported_languages[source_language]
                        
                        for i, text in enumerate(df[text_column]):
                            if pd.notna(text):
                                translated_text = processor.translate_text(str(text), src_lang_code, 'en')
                                translated_texts.append(translated_text)
                            else:
                                translated_texts.append("")
                            progress_bar.progress((i + 1) / len(df))
                        
                        df['translated_text'] = translated_texts
                        st.success("Translation completed!")
                        st.dataframe(df[['translated_text']].head())
                        
                        # Translation download
                        with st.expander("Download Translated Data"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("CSV (Translated)", key="single_trans_csv"):
                                    st.markdown(create_download_link(df, "translated_data", "CSV"), unsafe_allow_html=True)
                            with col2:
                                if st.button("Excel (Translated)", key="single_trans_excel"):
                                    st.markdown(create_download_link(df, "translated_data", "Excel"), unsafe_allow_html=True)
                            with col3:
                                if st.button("Text (Translated)", key="single_trans_text"):
                                    st.markdown(create_download_link(df, "translated_data", "Text"), unsafe_allow_html=True)
                        
                        # Keywords extraction
                        st.subheader("🔑 Step 3: Keyword Extraction")
                        if st.button("🔑 Extract and Translate Keywords", type="primary"):
                            all_text = ' '.join(df['translated_text'].dropna().astype(str))
                            keywords, keyword_freq = processor.extract_keywords(all_text, num_keywords)
                            
                            # Create keywords dataframe
                            keywords_df = pd.DataFrame({
                                'keyword': keywords,
                                'frequency': [keyword_freq[word] for word in keywords]
                            })
                            
                            # Translate keywords back
                            target_lang_code = processor.supported_languages[target_language_back]
                            translated_keywords = []
                            
                            progress_bar = st.progress(0)
                            for i, keyword in enumerate(keywords):
                                translated_keyword = processor.translate_text(keyword, 'en', target_lang_code)
                                translated_keywords.append(translated_keyword)
                                progress_bar.progress((i + 1) / len(keywords))
                            
                            keywords_df['translated_keyword'] = translated_keywords
                            st.success("Keywords extracted and translated!")
                            st.dataframe(keywords_df)
                            
                            # Keywords download
                            with st.expander("Download Keywords Data"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("CSV (Keywords)", key="single_kw_csv"):
                                        st.markdown(create_download_link(keywords_df, "keywords_data", "CSV"), unsafe_allow_html=True)
                                with col2:
                                    if st.button("Excel (Keywords)", key="single_kw_excel"):
                                        st.markdown(create_download_link(keywords_df, "keywords_data", "Excel"), unsafe_allow_html=True)
                                with col3:
                                    if st.button("Text (Keywords)", key="single_kw_text"):
                                        st.markdown(create_download_link(keywords_df, "keywords_data", "Text"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
