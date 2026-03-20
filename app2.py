import os
import json
import torch
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ============================
# CONFIG
# ============================
MODEL_NAME = "silma-ai/SILMA-Kashif-2B-Instruct-v1.0"
CHROMA_PATH = "chroma_db"
OUTPUT_JSON = "output1.json"

# ============================
# LOAD EMBEDDINGS
# ============================
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ============================
# LOAD LLM (TRANSFORMERS)
# ============================
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to("cuda")

    return tokenizer, model

tokenizer, model = load_llm()
embedder = load_embedder()

# ============================
# LLM COMPLETION
# ============================
def get_completion(prompt: str, max_tokens=512) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()
# ============================
# LOAD & INDEX JSON
# ============================
@st.cache_resource
def load_vector_db():
    if not os.path.exists(OUTPUT_JSON):
        st.error(f"❌ لم يتم العثور على {OUTPUT_JSON}")
        st.stop()

    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    if isinstance(data, dict) and "chunks" in data:
        meta = data.get("metadata", {})
        source_default = meta.get("source", "unknown")

        for chunk in data["chunks"]:
            docs.append(
                Document(
                    page_content=chunk.get("text", ""),
                    metadata={
                        "source": chunk.get("source", source_default),
                        "chunk_id": chunk.get("id"),
                        "language": chunk.get("language")
                    }
                )
            )

    elif isinstance(data, list):
        for item in data:
            docs.append(
                Document(
                    page_content=item.get("text", ""),
                    metadata={
                        "source": item.get("source", "unknown"),
                        "language": item.get("language")
                    }
                )
            )
    else:
        st.error("❌ شكل JSON غير مدعوم")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_PATH
    )

    try:
        vectordb.persist()
    except Exception:
        pass

    return vectordb

vectordb = load_vector_db()

# ============================
# RAG SEARCH FORMATTER
# ============================
def populate_rag_query(query, n_results=3):
    results = vectordb.similarity_search(query, k=n_results)
    result_str = ""

    for i, doc in enumerate(results):
        result_str += f"""
<SEARCH_RESULT>
<DOCUMENT>{doc.page_content}</DOCUMENT>
<SOURCE>{doc.metadata.get("source")}</SOURCE>
<CHUNK_ID>{doc.metadata.get("chunk_id", i)}</CHUNK_ID>
</SEARCH_RESULT>
"""
    return result_str

# ============================
# RAG PROMPT
# ============================
def make_rag_prompt(query, result_str):
    return f"""
أجب على السؤال باستخدام النتائج فقط.
إذا لم تجد الجواب في النتائج، قل: لا يوجد معلومات كافية.
اذكر المصدر في النهاية.

السؤال:
{query}

النتائج:
{result_str}

الإجابة:
"""

# ============================
# CQR
# ============================
def rewrite_query(query, chat_history):
    prompt = f"""
أعد صياغة السؤال اعتماداً على المحادثة السابقة:

{chat_history}

السؤال:
{query}

الصيغة الجديدة:
"""
    return get_completion(prompt)

# ============================
# HyDE
# ============================
def make_hyde_prompt(query):
    return f"""
أنشئ إجابة افتراضية قصيرة لهذا السؤال لاستخدامها في البحث:

{query}
"""

def answer_with_hyde(query):
    hyde_answer = get_completion(make_hyde_prompt(query))
    result_str = populate_rag_query(hyde_answer)
    return get_completion(make_rag_prompt(query, result_str))

# ============================
# CQR + RAG
# ============================
def perform_cqr_rag(query, chat_history):
    refined = rewrite_query(query, chat_history)
    result_str = populate_rag_query(refined)
    return refined, get_completion(make_rag_prompt(refined, result_str))

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="Local RAG + HyDE + CQR", layout="wide")
st.title("🧠 Faqiah")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

user_query = st.text_input("اكتب سؤالك:")
mode = st.selectbox("اختر نمط المعالجة:", ["RAG", "CQR + RAG", "HyDE + RAG"])

if st.button("إجابة"):
    if not user_query.strip():
        st.warning("❗ أدخل سؤالًا")
    else:
        if mode == "RAG":
            answer = get_completion(
                make_rag_prompt(user_query, populate_rag_query(user_query))
            )

        elif mode == "CQR + RAG":
            refined, answer = perform_cqr_rag(
                user_query,
                st.session_state.chat_history
            )
            st.info(f"🔁 السؤال بعد CQR: {refined}")

        else:
            answer = answer_with_hyde(user_query)

        st.session_state.chat_history += f"user: {user_query}\nassistant: {answer}\n"
        st.success(answer)
