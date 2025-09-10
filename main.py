import os
import io
import tempfile
import atexit
import json
import streamlit as st
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text
from dotenv import load_dotenv
import tiktoken

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import time

# ENV CONFIG AND SETUP
# ENV CONFIG AND SETUP
@st.cache_data
def load_environment() -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Load and validate environment variables"""
    try:
        load_dotenv()
        API_KEY = os.getenv("OPENAI_API_KEY")
        SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        DRIVE_MAIN_FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')

        missing_vars = [var for var, val in zip([
            "OPENAI_API_KEY", "GOOGLE_SERVICE_ACCOUNT_JSON", "GOOGLE_DRIVE_FOLDER_ID"],
            [API_KEY, SERVICE_ACCOUNT_JSON, DRIVE_MAIN_FOLDER_ID]
        ) if not val]

        if missing_vars:
            return None, f"Missing environment variables: {', '.join(missing_vars)}"

        return {
            "API_KEY": API_KEY,
            "SERVICE_ACCOUNT_JSON": SERVICE_ACCOUNT_JSON,
            "DRIVE_MAIN_FOLDER_ID": DRIVE_MAIN_FOLDER_ID
        }, None
    except Exception as e:
        return None, f"Failed to load environment variables: {str(e)}"


# Load environment variables
env_vars, env_error = load_environment()
if env_error:
    st.error(env_error)
    st.stop()

API_KEY = env_vars["API_KEY"]
SERVICE_ACCOUNT_JSON = env_vars["SERVICE_ACCOUNT_JSON"]
DRIVE_MAIN_FOLDER_ID = env_vars["DRIVE_MAIN_FOLDER_ID"]

# Model configurations
MODEL_MAP = {
    "Alpha - Fastest": "chatgpt-4o-latest",
    "Beta - Advanced reasoning & speed": "gpt-5",
    "Gamma - Large input capacity, detailed tasks": "gpt-4.1"
}

PRESET_QUERIES = {
    "Ask Bhagvad Gita": "According to Gita, use advise of Krishna to Arjun for similar cases and advise me accordingly. Please mention relevant shaloka and give short and crisp answer.",
    "Personalized Therapy Recommendation": "According to Psychotherapy guide book, please advise the correct therapy.",
    "Ancient Self": " According to CharakSamhita, please advise the correct therapy."
}

MODEL_CONFIGS = {
    "chatgpt-4o-latest": {
        "CONTEXT_CHUNKS": 8,            # Fits under 30k TPM
        "CHUNK_CHAR_LIMIT": 2000,       # Smaller per chunk
        "PROPOSAL_CHAR_LIMIT": 30000,
        "TOKEN_BUDGET": 128000,         # Model max, but our call stays < 30k
        "MAX_RESPONSE_TOKENS": 4000,    # Leaves room for input within TPM
        "SUMMARY_MAX_TOKENS": 1000,
    },
    "gpt-5": {
        "CONTEXT_CHUNKS": 8,
        "CHUNK_CHAR_LIMIT": 2000,
        "PROPOSAL_CHAR_LIMIT": 30000,
        "TOKEN_BUDGET": 400000,
        "MAX_RESPONSE_TOKENS": 4000,    # Keep total < 30k TPM
        "SUMMARY_MAX_TOKENS": 1000,
    },
    "gpt-4.1": {
        "CONTEXT_CHUNKS": 8,
        "CHUNK_CHAR_LIMIT": 2000,
        "PROPOSAL_CHAR_LIMIT": 30000,
        "TOKEN_BUDGET": 1000000,
        "MAX_RESPONSE_TOKENS": 4000,
        "SUMMARY_MAX_TOKENS": 1000,
    },
}


def error_handler(msg: str) -> None:
    st.error("An internal error occurred while processing your request. Please try again later.")

# Initialize OpenAI client
@st.cache_resource
def get_openai_client() -> Optional[OpenAI]:
    try:
        return OpenAI(api_key=API_KEY)
    except Exception as e:
        error_handler("")
        return None


client = get_openai_client()
if not client:
    st.stop()


def count_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Count tokens in text"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            st.warning("An internal error occurred while processing your request. Please try again later.")
            return len(text) // 4

    try:
        return len(enc.encode(text))
    except Exception as e:
        st.warning("An internal error occurred while processing your request. Please try again later.")
        return len(text) // 4


@st.cache_resource
def get_drive_service() -> Any:
    """Initialize Google Drive service"""
    try:
        parsed_json = json.loads(SERVICE_ACCOUNT_JSON)
        parsed_json["private_key"] = parsed_json["private_key"].replace("\\n", "\n")

        temp_service_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
        json.dump(parsed_json, temp_service_file)
        temp_service_file.close()

        SERVICE_ACCOUNT_FILE = temp_service_file.name
        atexit.register(lambda: os.remove(SERVICE_ACCOUNT_FILE) if os.path.exists(SERVICE_ACCOUNT_FILE) else None)

        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=['https://www.googleapis.com/auth/drive.readonly'],
        )
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        error_handler("")
        raise


@st.cache_data
def list_subfolders(parent_id):
    """List subfolders in Google Drive"""
    try:
        service = get_drive_service()
        results = []
        page_token = None

        while True:
            resp = service.files().list(
                q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="nextPageToken, files(id, name)",
                pageToken=page_token
            ).execute()

            results.extend(resp.get('files', []))
            page_token = resp.get('nextPageToken', None)
            if not page_token:
                break

        return results
    except Exception as e:
        st.error(f"Could not list subfolders: {str(e)}")
        return []


@st.cache_data
def list_txt_in_folder(folder_id):
    """List text files in a Google Drive folder"""
    try:
        service = get_drive_service()
        results = []
        page_token = None

        while True:
            resp = service.files().list(
                q=f"'{folder_id}' in parents and mimeType='text/plain' and trashed=false",
                pageSize=500,
                fields="nextPageToken, files(id, name)",
                pageToken=page_token
            ).execute()

            results.extend(resp.get('files', []))
            page_token = resp.get('nextPageToken', None)
            if not page_token:
                break

        return results
    except Exception as e:
        st.error(f"Could not list TXT files in folder: {str(e)}")
        return []


@st.cache_data
def download_txt_as_text(file_id):
    """Download text file from Google Drive"""
    try:
        service = get_drive_service()
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        max_retries = 5
        retries = 0
        while not done:
            try:
                _, done = downloader.next_chunk()
            except Exception as e:
                retries += 1
                if retries < max_retries:
                    st.warning(f"Download chunk error: {str(e)}. Retrying {retries}/{max_retries}...")
                else:
                    st.error(f"Download failed after {max_retries} retries: {str(e)}")
                    break

        fh.seek(0)
        try:
            return fh.read().decode('utf-8', errors='ignore')
        except Exception as e:
            st.error(f"Failed to decode downloaded text: {str(e)}")
            return ""
    except Exception as e:
        st.error(f"Failed to download text from drive: {str(e)}")
        return ""


def parse_uploaded_file(uploaded_file):
    """Parse uploaded file content"""
    try:
        fname = uploaded_file.name.lower()

        if fname.endswith('.txt'):
            try:
                bytes_content = uploaded_file.read()
                if isinstance(bytes_content, bytes):
                    return bytes_content.decode('utf-8', errors='ignore')
                else:
                    return str(bytes_content)
            except Exception as e:
                st.error(f"Failed to decode TXT file: {str(e)}. Please use UTF-8 encoding.")
                return ""

        elif fname.endswith('.docx'):
            try:
                doc = Document(io.BytesIO(uploaded_file.read()))
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.error(f"Failed to parse DOCX: {str(e)}. File may be corrupt or wrong format.")
                return ""

        elif fname.endswith('.pdf'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp.close()
                    text = extract_pdf_text(tmp.name)
                    os.remove(tmp.name)
                    return text
            except Exception as e:
                st.error(f"Failed to parse PDF: {str(e)}. The file may be encrypted or corrupt.")
                return ""
        else:
            st.warning("Unsupported file format. Only .txt, .docx, .pdf allowed.")
            return ""
    except Exception as e:
        st.error(f"Could not parse uploaded file: {str(e)}")
        return ""


def chunk_documents(reference_docs: List[str], chunk_size: int) -> List[Dict[str, Any]]:
    """Split documents into chunks"""
    try:
        chunks = []
        for doc_index, doc in enumerate(reference_docs):
            doc = (doc or "").strip()
            if not doc:
                continue
            for i in range(0, len(doc), chunk_size):
                chunk = doc[i:i + chunk_size]
                if chunk.strip():
                    chunks.append({
                        'text': chunk,
                        'doc_index': doc_index,
                        'chunk_start': i
                    })
        return chunks
    except Exception as e:
        error_handler("")
        return []


def safe_openai_call(fn: Any, *args, retries: int = 3, **kwargs) -> Any:
    """Safe OpenAI call with retries"""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                error_handler("")
                return None


def get_embeddings_for_chunks(chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Get embeddings for document chunks"""
    try:
        texts = [chunk['text'] for chunk in chunks]
        max_batch = 96
        out = []

        for i in range(0, len(texts), max_batch):
            response = safe_openai_call(
                client.embeddings.create,
                input=texts[i:i + max_batch],
                model="text-embedding-3-small"
            )

            if response is None or not hasattr(response, "data"):
                error_handler("")
                return []

            emb = [np.array(d.embedding) for d in response.data]
            out.extend(emb)

        return out
    except Exception as e:
        error_handler("")
        return []


def embedding_for_query(query):
    """Get embedding for query"""
    try:
        response = safe_openai_call(
            client.embeddings.create,
            input=[query],
            model="text-embedding-3-small"
        )

        if response is None or not hasattr(response, "data"):
            st.error("Failed to compute embedding for query via OpenAI.")
            return np.zeros(1536)

        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Embedding for query failed: {str(e)}")
        return np.zeros(1536)


def retrieve_relevant_chunks(reference_docs: List[str], user_query: str, k: int, chunk_size: int) -> List[str]:
    """Retrieve relevant chunks using semantic similarity"""
    try:
        # Create a hash for caching
        ref_hash = hash(tuple(reference_docs))

        if ("rag_ref_docs_hash" not in st.session_state or
                st.session_state.rag_ref_docs_hash != ref_hash):
            st.session_state.rag_chunks = chunk_documents(reference_docs, chunk_size)
            st.session_state.rag_chunks_embeddings = (
                get_embeddings_for_chunks(st.session_state.rag_chunks)
                if st.session_state.rag_chunks else []
            )
            st.session_state.rag_ref_docs_hash = ref_hash

        if not st.session_state.rag_chunks:
            return []

        query_emb = embedding_for_query(user_query)
        chunk_embs = st.session_state.rag_chunks_embeddings

        if not chunk_embs:
            st.warning("No embeddings for context. Try reloading reference docs.")
            return []

        # Batch similarity calculation using numpy for speed
        chunk_embs_np = np.stack(chunk_embs)
        query_emb_np = np.array(query_emb)
        sims = np.dot(chunk_embs_np, query_emb_np) / (np.linalg.norm(chunk_embs_np, axis=1) * np.linalg.norm(query_emb_np) + 1e-8)

        # Get top k chunks
        idxs = np.argsort(sims)[::-1][:k]
        relevant_chunks = [st.session_state.rag_chunks[i]['text'] for i in idxs]

        return relevant_chunks
    except Exception as e:
        error_handler("")
        return []


def assemble_context(reference_docs: List[str], user_query: str, k: int, chunk_size: int) -> str:
    """Assemble context from relevant chunks"""
    try:
        relevant_chunks = retrieve_relevant_chunks(reference_docs, user_query, k, chunk_size)

        if not relevant_chunks:
            st.warning("No relevant reference documents found for this query.")

        context_block = "\n\n".join(relevant_chunks)
        return context_block
    except Exception as e:
        error_handler("")
        return ""


def run_model(context_block: str, proposal_block: Optional[str], user_query: str, model_name: str, config: Dict[str, Any]) -> str:
    """Run the model with context and query"""
    use_proposal = (
        proposal_block and
        ("proposal" in user_query.lower() or "uploaded document" in user_query.lower())
    )

    prompt = f"""
You are an expert internal auditor and financial policy assistant. Using only the content provided (uploaded files, templates, and references), generate a clear, structured, and accurate report. Do not use external knowledge.

1. Tone & Purpose
Keep your tone professional, clear, and informative.
Reports should be understandable to non-experts but grounded in policy.

2. Structure
Use clear sections with bold titles, such as:
Summary: What the document is and what you're evaluating.
Compliance Review: List applicable rules (e.g., Rule 136, GFR 2017) and check adherence.
Observations: Note missing data, errors, or issues.
Conclusion/Recommendation: State whether the proposal is acceptable, needs correction, or is non-compliant.

3. Citations
When referencing policies:
Mention exact rule names and numbers.
Use parentheses or brackets for citations: e.g., (Rule 47, GFR 2017), [Audit Manual, Sec. 3.2].

4. Clarity & Formatting
Use bullet points where needed.
Avoid long paragraphs.
Be concise but complete.

5. Restrictions
Do not hallucinate information or rules.
Do not say "as per guidelines" without specifying the document.
Do not copy large sections of text—summarize instead.

Reference Documents:
{context_block}
""" + (f"\nProposal document:\n{proposal_block}\n" if use_proposal else "") + f"""
User Question:
{user_query}

If the answer is not found in the provided context, respond: "The answer is not present in the provided references." Otherwise, answer fully, using a friendly, complete, professional and helpful style.
"""

    try:
        input_tokens = count_tokens(prompt, model_name)

        if input_tokens > (config["TOKEN_BUDGET"] - config["MAX_RESPONSE_TOKENS"]):
            # Attempt truncation
            orig_chunks = context_block.split("\n\n")
            while (input_tokens > (config["TOKEN_BUDGET"] - config["MAX_RESPONSE_TOKENS"])
                   and len(orig_chunks) > 1):
                orig_chunks = orig_chunks[:-1]
                context_block_new = "\n\n".join(orig_chunks)
                prompt = prompt.replace(context_block, context_block_new)
                context_block = context_block_new
                input_tokens = count_tokens(prompt, model_name)

            if input_tokens > (config["TOKEN_BUDGET"] - config["MAX_RESPONSE_TOKENS"]):
                st.warning("Prompt too long. Try selecting fewer reference documents.")
                return "Error: Token limit exceeded."

        response = safe_openai_call(
            client.chat.completions.create,
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert auditor and policy assistant. Your job is to help the user by providing high-quality, easy to understand, fully structured answers using ONLY the context supplied."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["MAX_RESPONSE_TOKENS"],
        )

        if not response or not hasattr(response, "choices"):
            error_handler("")
            return "An error occurred in generating the response."

        return response.choices[0].message.content
    except Exception:
        error_handler("")
        return "Model run error."


def make_summary(full_answer, model_name, config):
    """Generate summary of the full answer"""
    summary_prompt = f"""
Summarize the following answer in 2-4 lines for a 'Summary (TL;DR)' box at the end of a report. Focus on only the essential points and avoid repetition. Write in clear plain English.

Answer:
{full_answer}

TL;DR:
"""
    try:
        response = safe_openai_call(
            client.chat.completions.create,
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who summarizes text for users in short, plain language for non-experts."
                },
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=config["SUMMARY_MAX_TOKENS"],
        )

        if not response or not hasattr(response, "choices"):
            return "Could not generate summary."

        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"Could not generate summary: {str(e)}")
        return "Could not generate summary."


# STREAMLIT UI
def main():
    st.set_page_config(page_title="Inner Equilibrium: Combining Bhagavad Gita Teachings with Modern Psychometrics", layout="wide")
    PASSWORD = os.getenv("PASSWORD")

    # Ask for password if not authenticated yet
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔐 Authentication Required")

        with st.form("login_form"):
            pwd = st.text_input("Enter password:", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login")

        if submit_button:
            if pwd == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        st.stop()

    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("Masthead.png", width='stretch')
        except Exception:
            pass
        st.markdown("<h3 style='text-align: center;'>Inner Equilibrium: Combining Bhagavad Gita Teachings with Modern Psychometrics (गीतमानस)</h3>",
                    unsafe_allow_html=True)

    # Hide Streamlit styling
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        .st-emotion-cache-13ln4jf {display: none;}
        .st-emotion-cache-ocqkz7 {display: none;}
        footer {visibility: hidden !important;}
        [data-testid="stToolbar"] {visibility: hidden !important;}
        .block-container { padding-top: 1rem !important; }
        .main { padding-top: 0rem !important; }
        </style>
    """
    try:
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    except Exception:
        pass

    # Initialize Google Drive service
    try:
        drive_service = get_drive_service()
    except Exception as e:
        st.error(f"Google Drive authentication/setup failed: {str(e)}")
        st.stop()

    # Subfolder selection
    st.subheader("Select Reference Subfolders")
    try:
        subfolders = list_subfolders(DRIVE_MAIN_FOLDER_ID)
        if not subfolders:
            st.warning("No subfolders found in project.")
            st.stop()

        subfolder_names = [f['name'] for f in subfolders]
        subfolder_map = {f['name']: f['id'] for f in subfolders}

        selected_subfolders = st.multiselect("Choose one or more subfolders", subfolder_names)
    except Exception as e:
        st.error(f"Listing subfolders failed: {str(e)}")
        st.stop()

    # Initialize reference docs in session state
    if "reference_docs" not in st.session_state:
        st.session_state.reference_docs = []

    # Fetch reference documents
    if selected_subfolders:
        if st.button(f"Fetch and combine files from: {', '.join(selected_subfolders)}"):
            docs = []
            with st.spinner("Fetching reference files..."):
                try:
                    for subfolder_name in selected_subfolders:
                        files_in_sub = list_txt_in_folder(subfolder_map[subfolder_name])
                        for file in files_in_sub:
                            raw = download_txt_as_text(file['id'])
                            if raw:
                                docs.append(raw.strip())

                    st.session_state.reference_docs = docs

                    # Reset RAG cache
                    for k in ['rag_chunks', 'rag_chunks_embeddings', 'rag_ref_docs_hash']:
                        if k in st.session_state:
                            del st.session_state[k]

                    st.success(f"Loaded {len(docs)} reference files.")
                except Exception as e:
                    st.error(f"Reference document fetching error: {str(e)}")
    else:
        st.info("Please select at least one subfolder to load reference documents.")

    reference_docs = st.session_state.reference_docs

    # Display reference docs status
    if reference_docs:
        st.info(f"Currently loaded reference documents: {len(reference_docs)}")
    else:
        st.info("No reference documents loaded.")

    # Model selection (moved up)
    st.subheader("Select Model")
    try:
        model_cols = st.columns(len(MODEL_MAP))
        for i, (label, model) in enumerate(MODEL_MAP.items()):
            if model_cols[i].button(label):
                st.session_state.selected_model = model
                st.session_state.selected_model_label = label
    except Exception:
        st.session_state.selected_model = "gpt-4.1"
        st.session_state.selected_model_label = "Gamma - Large input capacity, detailed tasks"

    # Initialize model selection if not exists
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4.1"
    if "selected_model_label" not in st.session_state:
        st.session_state.selected_model_label = "Gamma - Large input capacity, detailed tasks"

    st.success(f"Model selected: {st.session_state.selected_model_label}")
    selected_model = st.session_state.selected_model

    # Get model configuration
    config = MODEL_CONFIGS.get(selected_model, MODEL_CONFIGS["gpt-4.1"])

    # Mode selection
    st.subheader("Select Mode")
    mode = st.radio(
        "Choose mode",
        ["Report/Template Generation", "Query Answering"],
        horizontal=True
    )

    # Report/Template Generation Mode
    if mode == "Report/Template Generation":
        st.subheader("Report/Template Query")

        # Inputs for report generation (same as counselling)
        if "report_question" not in st.session_state:
            st.session_state.report_question = ""
        if "personality_score" not in st.session_state:
            st.session_state.personality_score = 0
        if "vedic_personality_score" not in st.session_state:
            st.session_state.vedic_personality_score = 0

        report_question = st.text_input(
            "Ask your Question for Counselling",
            value=st.session_state.get("report_question", ""),
            key="report_question_input_1"
        )
        if report_question != st.session_state.get("report_question", ""):
            st.session_state.report_question = report_question

        personality_score = st.number_input(
            "Your Personality Score",
            min_value=0, max_value=100, value=st.session_state.get("personality_score", 0), step=1,
            key="personality_score_input_1"
        )
        st.session_state.personality_score = personality_score

        vedic_personality_score = st.number_input(
            "Vedic Personality Score",
            min_value=0, max_value=100, value=st.session_state.get("vedic_personality_score", 0), step=1,
            key="vedic_personality_score_input_1"
        )
        st.session_state.vedic_personality_score = vedic_personality_score

        # Quick prompts at the end
        st.subheader("Quick Prompts")
        if "quick_prompt" not in st.session_state:
            st.session_state.quick_prompt = None

        quick_col1, quick_col2, quick_col3 = st.columns(3)
        try:
            with quick_col1:
                if st.button("Ask Bhagvad Gita"):
                    st.session_state.quick_prompt = PRESET_QUERIES["Ask Bhagvad Gita"]
            with quick_col2:
                if st.button("Personalized Therapy Recommendation"):
                    st.session_state.quick_prompt = PRESET_QUERIES["Personalized Therapy Recommendation"]
            with quick_col3:
                if st.button("Ancient Self"):
                    st.session_state.quick_prompt = PRESET_QUERIES["Ancient Self"]
        except Exception:
            pass

        #submit_custom_query = st.button("Submit")  # COMMENTED OUT
        submit_custom_query = False  # Always False since button is commented

        # Handle query submission
        used_query = report_question
        if st.session_state.get("quick_prompt"):
            used_query = st.session_state.quick_prompt

        # Only run if a quick prompt is selected (since submit is commented)
        if st.session_state.get("quick_prompt") and used_query:
            if reference_docs:
                # All three fields are optional, so build extra_info accordingly
                extra_info = ""
                if personality_score:
                    extra_info += f"\nPersonality Score: {personality_score}"
                if vedic_personality_score:
                    extra_info += f"\nVedic Personality Score: {vedic_personality_score}"

                with st.spinner("Fetching relevant context and generating report..."):
                    try:
                        context_block = assemble_context(
                            reference_docs,
                            (used_query or "") + extra_info,
                            config["CONTEXT_CHUNKS"],
                            config["CHUNK_CHAR_LIMIT"]
                        )
                        output = run_model(context_block, None, (used_query or "") + extra_info, selected_model, config)
                        summary = make_summary(output, selected_model, config)
                    except Exception as e:
                        st.error(f"Report generation error: {str(e)}")
                        output = "Error generating report"
                        summary = "Error"

                # Display results
                st.subheader("Result")
                try:
                    st.write(output)
                    st.markdown(f"---\n<b>Summary (TL;DR):</b><br>{summary}", unsafe_allow_html=True)

                    # Download button
                    download_content = output + "\n\nSummary (TL;DR):\n" + summary
                    st.download_button(
                        "Download response as TXT",
                        download_content,
                        file_name="audit_response.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error displaying results: {str(e)}")

                # Reset quick prompt
                st.session_state.quick_prompt = None
            else:
                st.info("Please select and load reference documents from Google Drive.")

    # Model selection
    st.subheader("Select Model")
    try:
        model_cols = st.columns(len(MODEL_MAP))
        for i, (label, model) in enumerate(MODEL_MAP.items()):
            if model_cols[i].button(label):
                st.session_state.selected_model = model
                st.session_state.selected_model_label = label
    except Exception:
        st.session_state.selected_model = "gpt-4.1"
        st.session_state.selected_model_label = "Gamma - Large input capacity, detailed tasks"

    # Initialize model selection if not exists
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4.1"
    if "selected_model_label" not in st.session_state:
        st.session_state.selected_model_label = "Gamma - Large input capacity, detailed tasks"

    st.success(f"Model selected: {st.session_state.selected_model_label}")
    selected_model = st.session_state.selected_model

    # Get model configuration
    config = MODEL_CONFIGS.get(selected_model, MODEL_CONFIGS["gpt-4.1"])

    # Query Answering Mode
    if mode == "Query Answering":
        st.subheader("Ask your Question for Counselling")

        # Quick prompts for Query Answering mode
        if "qa_quick_prompt" not in st.session_state:
            st.session_state.qa_quick_prompt = None

        qa_quick_col1, qa_quick_col2, qa_quick_col3 = st.columns(3)
        try:
            with qa_quick_col1:
                if st.button("Ask Bhagvad Gita", key="qa_gita"):
                    st.session_state.qa_quick_prompt = PRESET_QUERIES["Ask Bhagvad Gita"]
            with qa_quick_col2:
                if st.button("Personalized Therapy Recommendation", key="qa_therapy"):
                    st.session_state.qa_quick_prompt = PRESET_QUERIES["Personalized Therapy Recommendation"]
            with qa_quick_col3:
                if st.button("Ancient Self", key="qa_ancient"):
                    st.session_state.qa_quick_prompt = PRESET_QUERIES["Ancient Self"]
        except Exception:
            pass

        # Inputs for counselling
        if "counselling_question" not in st.session_state:
            st.session_state.counselling_question = ""
        if "personality_score" not in st.session_state:
            st.session_state.personality_score = 0
        if "vedic_personality_score" not in st.session_state:
            st.session_state.vedic_personality_score = 0

        # If a quick prompt is selected, use it as the question
        if st.session_state.get("qa_quick_prompt"):
            counselling_question = st.session_state.qa_quick_prompt
        else:
            counselling_question = st.text_input(
                "Ask your Question for Counselling",
                value=st.session_state.get("counselling_question", ""),
                key="counselling_question_input_1"
            )
            if counselling_question != st.session_state.get("counselling_question", ""):
                st.session_state.counselling_question = counselling_question

        personality_score = st.number_input(
            "Your Personality Score",
            min_value=0, max_value=100, value=st.session_state.get("personality_score", 0), step=1,
            key="personality_score_input_3"
        )
        st.session_state.personality_score = personality_score

        vedic_personality_score = st.number_input(
            "Vedic Personality Score",
            min_value=0, max_value=100, value=st.session_state.get("vedic_personality_score", 0), step=1,
            key="vedic_personality_score_input_3"
        )
        st.session_state.vedic_personality_score = vedic_personality_score

        submit_counselling = st.button("Get Counselling Answer")

        # Use the quick prompt if set, otherwise use the text area value
        used_counselling_question = st.session_state.get("qa_quick_prompt") or counselling_question

        if submit_counselling and used_counselling_question:
            if reference_docs:
                with st.spinner("Searching references and answering..."):
                    try:
                        extra_info = f"\nPersonality Score: {personality_score}\nVedic Personality Score: {vedic_personality_score}"
                        context_block = assemble_context(
                            reference_docs,
                            used_counselling_question + extra_info,
                            config["CONTEXT_CHUNKS"],
                            config["CHUNK_CHAR_LIMIT"]
                        )
                        output = run_model(context_block, None, used_counselling_question + extra_info, selected_model, config)
                        summary = make_summary(output, selected_model, config)
                    except Exception as e:
                        st.error(f"QA error: {str(e)}")
                        output = "Error generating answer"
                        summary = "Error"

                # Display results
                st.subheader("Answer")
                try:
                    st.write(output)
                    st.markdown(f"---\n<b>Summary (TL;DR):</b><br>{summary}", unsafe_allow_html=True)

                    # Download button
                    download_content = output + "\n\nSummary (TL;DR):\n" + summary
                    st.download_button(
                        "Download answer as TXT",
                        download_content,
                        file_name="counselling_answer.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error displaying answer: {str(e)}")
                # Reset quick prompt after use
                st.session_state.qa_quick_prompt = None
            else:
                st.info("Please select and load reference documents from Google Drive.")

if __name__ == "__main__":
    main()