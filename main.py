import os
import io
import tempfile
import atexit
import json
import streamlit as st
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import time


# ENV CONFIG AND SETUP
@st.cache_data
def load_environment() -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Load and validate environment variables"""
    try:
        load_dotenv()
        API_KEY = os.getenv("OPENAI_API_KEY")
        SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        DRIVE_MAIN_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

        missing_vars = [
            var
            for var, val in zip(
                [
                    "OPENAI_API_KEY",
                    "GOOGLE_SERVICE_ACCOUNT_JSON",
                    "GOOGLE_DRIVE_FOLDER_ID",
                ],
                [API_KEY, SERVICE_ACCOUNT_JSON, DRIVE_MAIN_FOLDER_ID],
            )
            if not val
        ]

        if missing_vars:
            return None, f"Missing environment variables: {', '.join(missing_vars)}"

        return {
            "API_KEY": API_KEY,
            "SERVICE_ACCOUNT_JSON": SERVICE_ACCOUNT_JSON,
            "DRIVE_MAIN_FOLDER_ID": DRIVE_MAIN_FOLDER_ID,
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
    "Gamma - Large input capacity, detailed tasks": "gpt-4.1",
}

PRESET_QUERIES = {
    "Ask Bhagvad Gita": "According to Gita, use advise of Krishna to Arjun for similar cases and advise me accordingly. Please mention relevant shaloka and give short and crisp answer.",
    "Personalized Therapy Recommendation": "According to Psychotherapy guide book, please advise the correct therapy.",
    "Ancient Self": "According to CharakSamhita, please advise the correct therapy.",
}

MODEL_CONFIGS = {
    "chatgpt-4o-latest": {
        "CONTEXT_CHUNKS": 8,
        "CHUNK_CHAR_LIMIT": 2000,
        "TOKEN_BUDGET": 128000,
        "MAX_RESPONSE_TOKENS": 4000,
        "SUMMARY_MAX_TOKENS": 1000,
    },
    "gpt-5": {
        "CONTEXT_CHUNKS": 8,
        "CHUNK_CHAR_LIMIT": 2000,
        "TOKEN_BUDGET": 400000,
        "MAX_RESPONSE_TOKENS": 4000,
        "SUMMARY_MAX_TOKENS": 1000,
    },
    "gpt-4.1": {
        "CONTEXT_CHUNKS": 8,
        "CHUNK_CHAR_LIMIT": 2000,
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

        temp_service_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
        json.dump(parsed_json, temp_service_file)
        temp_service_file.close()

        SERVICE_ACCOUNT_FILE = temp_service_file.name
        atexit.register(
            lambda: (
                os.remove(SERVICE_ACCOUNT_FILE)
                if os.path.exists(SERVICE_ACCOUNT_FILE)
                else None
            )
        )

        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        return build("drive", "v3", credentials=creds)
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
            resp = (
                service.files()
                .list(
                    q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )

            results.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken", None)
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
            resp = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and mimeType='text/plain' and trashed=false",
                    pageSize=500,
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )

            results.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken", None)
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
            return fh.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Failed to decode downloaded text: {str(e)}")
            return ""
    except Exception as e:
        st.error(f"Failed to download text from drive: {str(e)}")
        return ""


def safe_openai_call(fn: Any, *args, retries: int = 3, **kwargs) -> Any:
    """Safe OpenAI call with retries"""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                error_handler("")
                return None


def chunk_documents(documents: List[str], chunk_size: int) -> List[Dict[str, Any]]:
    """Chunk documents into smaller pieces"""
    chunks = []
    for doc_idx, doc in enumerate(documents):
        for i in range(0, len(doc), chunk_size):
            chunks.append({
                "text": doc[i:i + chunk_size],
                "doc_idx": doc_idx,
                "chunk_idx": len(chunks)
            })
    return chunks


def get_embeddings_for_chunks(chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Get embeddings for document chunks"""
    try:
        texts = [chunk["text"] for chunk in chunks]
        max_batch = 96
        out = []

        for i in range(0, len(texts), max_batch):
            response = safe_openai_call(
                client.embeddings.create,
                input=texts[i : i + max_batch],
                model="text-embedding-3-small",
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
            client.embeddings.create, input=[query], model="text-embedding-3-small"
        )

        if response is None or not hasattr(response, "data"):
            st.error("Failed to compute embedding for query via OpenAI.")
            return np.zeros(1536)

        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Embedding for query failed: {str(e)}")
        return np.zeros(1536)


def retrieve_relevant_chunks(
    reference_docs: List[str], user_query: str, k: int, chunk_size: int
) -> List[str]:
    """Retrieve relevant chunks using semantic similarity"""
    try:
        # Create a hash for caching
        ref_hash = hash(tuple(reference_docs))

        if (
            "rag_ref_docs_hash" not in st.session_state
            or st.session_state.rag_ref_docs_hash != ref_hash
        ):
            st.session_state.rag_chunks = chunk_documents(reference_docs, chunk_size)
            st.session_state.rag_chunks_embeddings = (
                get_embeddings_for_chunks(st.session_state.rag_chunks)
                if st.session_state.rag_chunks
                else []
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
        sims = np.dot(chunk_embs_np, query_emb_np) / (
            np.linalg.norm(chunk_embs_np, axis=1) * np.linalg.norm(query_emb_np) + 1e-8
        )

        # Get top k chunks
        idxs = np.argsort(sims)[::-1][:k]
        relevant_chunks = [st.session_state.rag_chunks[i]["text"] for i in idxs]

        return relevant_chunks
    except Exception as e:
        error_handler("")
        return []


def assemble_context(
    reference_docs: List[str], user_query: str, k: int, chunk_size: int
) -> str:
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


def run_model(
    context_block: str,
    user_query: str,
    model_name: str,
    config: Dict[str, Any],
    is_detailed: bool = True
) -> str:
    """Run the model with context and query"""
    
    # Different prompts for detailed vs brief answers
    if is_detailed:
        system_prompt = """You are an expert spiritual counselor combining Bhagavad Gita teachings with modern psychometrics. Using only the content provided (uploaded files and references), generate a clear, structured, and comprehensive counseling response.

1. Tone & Purpose: Keep your tone warm, empathetic, and spiritually grounding.
2. Structure: Use clear sections with bold titles for comprehensive guidance.
3. Citations: When referencing texts, mention exact verse numbers and sources.
4. Formatting: Use bullet points and clear paragraphs for readability.
5. Restrictions: Do not use external knowledge - only use provided references."""
        
        max_tokens = config["MAX_RESPONSE_TOKENS"]
    else:
        system_prompt = """You are a spiritual counselor providing brief, focused guidance based on Bhagavad Gita teachings and modern psychometrics. Using only the provided references, give concise but meaningful advice.

Keep responses short and to the point while maintaining warmth and spiritual wisdom. Focus on the most essential guidance only."""
        
        max_tokens = min(config["MAX_RESPONSE_TOKENS"] // 2, 2000)  # Shorter for brief answers

    prompt = f"""
Reference Documents:
{context_block}

User Question:
{user_query}

If the answer is not found in the provided context, respond: "The answer is not present in the provided references." Otherwise, provide {'comprehensive' if is_detailed else 'brief'} guidance using a warm, empathetic, and spiritually grounded approach.
"""

    try:
        input_tokens = count_tokens(prompt, model_name)

        if input_tokens > (config["TOKEN_BUDGET"] - max_tokens):
            # Attempt truncation
            orig_chunks = context_block.split("\n\n")
            while (
                input_tokens > (config["TOKEN_BUDGET"] - max_tokens)
                and len(orig_chunks) > 1
            ):
                orig_chunks = orig_chunks[:-1]
                context_block_new = "\n\n".join(orig_chunks)
                prompt = prompt.replace(context_block, context_block_new)
                context_block = context_block_new
                input_tokens = count_tokens(prompt, model_name)

            if input_tokens > (config["TOKEN_BUDGET"] - max_tokens):
                st.warning("Prompt too long. Try selecting fewer reference documents.")
                return "Error: Token limit exceeded."

        response = safe_openai_call(
            client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
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
Summarize the following spiritual counseling answer in 2-4 lines for a 'Summary (TL;DR)' box. Focus on the essential spiritual guidance and practical steps. Write in clear, compassionate language.

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
                    "content": "You are a helpful assistant who summarizes spiritual guidance in short, compassionate language.",
                },
                {"role": "user", "content": summary_prompt},
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
    st.set_page_config(
        page_title="Inner Equilibrium: Combining Bhagavad Gita Teachings with Modern Psychometrics",
        layout="wide",
    )
    PASSWORD = os.getenv("PASSWORD")

    # Authentication
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
            st.image("Masthead.png", width="stretch")
        except Exception:
            pass
        st.markdown(
            "<h3 style='text-align: center;'>Inner Equilibrium: Combining Bhagavad Gita Teachings with Modern Psychometrics (गीतमानस)</h3>",
            unsafe_allow_html=True,
        )

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

        subfolder_names = [f["name"] for f in subfolders]
        subfolder_map = {f["name"]: f["id"] for f in subfolders}

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
                            raw = download_txt_as_text(file["id"])
                            if raw:
                                docs.append(raw.strip())

                    st.session_state.reference_docs = docs

                    # Reset RAG cache
                    for k in ["rag_chunks", "rag_chunks_embeddings", "rag_ref_docs_hash"]:
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

    # Model selection
    st.subheader("Select Model")
    model_cols = st.columns(len(MODEL_MAP))
    for i, (label, model) in enumerate(MODEL_MAP.items()):
        if model_cols[i].button(label):
            st.session_state.selected_model = model
            st.session_state.selected_model_label = label

    # Initialize model selection if not exists
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4.1"
        st.session_state.selected_model_label = "Gamma - Large input capacity, detailed tasks"

    st.success(f"Model selected: {st.session_state.selected_model_label}")
    selected_model = st.session_state.selected_model
    config = MODEL_CONFIGS.get(selected_model, MODEL_CONFIGS["gpt-4.1"])

    # Mode selection
    st.subheader("Select Mode")
    mode = st.radio(
        "Choose mode",
        ["Report Generation (Detailed)", "Query Answering (Brief)"],
        horizontal=True,
    )


    # Only show input fields and quick prompts for Report Generation (Detailed) mode
    is_detailed = "Detailed" in mode
    if is_detailed:
        # All three as text fields
        if "user_question" not in st.session_state:
            st.session_state.user_question = ""
        user_question = st.text_input(
            "Counselling Question",
            value=st.session_state.get("user_question", ""),
            placeholder="Ask your question for spiritual guidance..."
        )
        st.session_state.user_question = user_question

        if "personality_score" not in st.session_state:
            st.session_state.personality_score = ""
        personality_score = st.text_input(
            "Your Personality Score",
            value=st.session_state.get("personality_score", ""),
            placeholder="Enter your personality score..."
        )
        st.session_state.personality_score = personality_score

        if "vedic_personality_score" not in st.session_state:
            st.session_state.vedic_personality_score = ""
        vedic_personality_score = st.text_input(
            "Vedic Personality Score",
            value=st.session_state.get("vedic_personality_score", ""),
            placeholder="Enter your vedic personality score..."
        )
        st.session_state.vedic_personality_score = vedic_personality_score

        # Quick prompts (only for detailed mode)
        st.subheader("Quick Prompts")
        if "selected_quick_prompt" not in st.session_state:
            st.session_state.selected_quick_prompt = None

        quick_prompt_labels = list(PRESET_QUERIES.keys())
        quick_prompt_backend = PRESET_QUERIES
        quick_cols = st.columns(3)
        for i, label in enumerate(quick_prompt_labels):
            btn_style = "background-color:#4CAF50;color:white;" if st.session_state.selected_quick_prompt == label else ""
            with quick_cols[i]:
                if st.button(label, key=f"quick_{label}", help=None):
                    st.session_state.selected_quick_prompt = label

        # Add custom CSS for selected button
        st.markdown("""
            <style>
            div[data-testid="column"] button[style*='background-color:#4CAF50'] {
                border: 2px solid #388e3c !important;
                font-weight: bold !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Show confirmation message for selected quick prompt
        if st.session_state.selected_quick_prompt:
            selected_label = st.session_state.selected_quick_prompt
            selected_query = quick_prompt_backend[selected_label]
            st.success(f"Quick prompt selected: {selected_label}")

        # Add a paragraph field and submit button for custom prompt
        if "custom_report_prompt" not in st.session_state:
            st.session_state.custom_report_prompt = ""
        custom_report_prompt = st.text_area(
            "Or type your own detailed prompt",
            value=st.session_state.get("custom_report_prompt", ""),
            placeholder="Type your custom prompt for detailed report..."
        )
        st.session_state.custom_report_prompt = custom_report_prompt

        if st.button("Generate Detailed Report"):
            # Use custom prompt if provided, else use quick prompt
            if custom_report_prompt.strip():
                query_to_use = custom_report_prompt.strip()
            elif st.session_state.selected_quick_prompt:
                query_to_use = quick_prompt_backend[st.session_state.selected_quick_prompt]
            else:
                query_to_use = ""

            if not query_to_use:
                st.warning("Please select a quick prompt or type a custom prompt.")
            elif reference_docs:
                extra_info = ""
                if personality_score:
                    extra_info += f"\nPersonality Score: {personality_score}"
                if vedic_personality_score:
                    extra_info += f"\nVedic Personality Score: {vedic_personality_score}"
                with st.spinner("Generating detailed report..."):
                    try:
                        context_block = assemble_context(
                            reference_docs,
                            query_to_use + extra_info,
                            config["CONTEXT_CHUNKS"],
                            config["CHUNK_CHAR_LIMIT"],
                        )
                        output = run_model(
                            context_block,
                            query_to_use + extra_info,
                            selected_model,
                            config,
                            is_detailed=True
                        )
                        summary = make_summary(output, selected_model, config)
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        output = "Error generating response"
                        summary = "Error"

                # Display results
                st.subheader("Result")
                st.write(output)
                if summary:
                    st.markdown(f"---\n<b>Summary (TL;DR):</b><br>{summary}", unsafe_allow_html=True)

                # Download button
                download_content = output
                if summary:
                    download_content += f"\n\nSummary (TL;DR):\n{summary}"
                st.download_button(
                    "Download response as TXT",
                    download_content,
                    file_name="counselling_report.txt",
                    mime="text/plain",
                )
            else:
                st.info("Please select and load reference documents from Google Drive first.")
    else:
        # Brief mode: show a text field for the user's question
        submit_label = "Get Brief Answer"
        if "brief_user_question" not in st.session_state:
            st.session_state.brief_user_question = ""
        brief_user_question = st.text_area(
            "Your Question",
            value=st.session_state.get("brief_user_question", ""),
            placeholder="Ask your question for spiritual guidance..."
        )
        st.session_state.brief_user_question = brief_user_question

        if st.button(submit_label) and brief_user_question.strip():
            if reference_docs:
                with st.spinner("Getting brief answer..."):
                    try:
                        context_block = assemble_context(
                            reference_docs,
                            brief_user_question,
                            config["CONTEXT_CHUNKS"],
                            config["CHUNK_CHAR_LIMIT"],
                        )
                        output = run_model(
                            context_block,
                            brief_user_question,
                            selected_model,
                            config,
                            is_detailed=False
                        )
                        summary = None
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        output = "Error generating response"
                        summary = "Error"

                # Display results
                st.subheader("Result")
                st.write(output)
                if summary:
                    st.markdown(f"---\n<b>Summary (TL;DR):</b><br>{summary}", unsafe_allow_html=True)

                # Download button
                download_content = output
                if summary:
                    download_content += f"\n\nSummary (TL;DR):\n{summary}"
                st.download_button(
                    "Download response as TXT",
                    download_content,
                    file_name="counselling_answer.txt",
                    mime="text/plain",
                )
            else:
                st.info("Please select and load reference documents from Google Drive first.")


if __name__ == "__main__":
    main()