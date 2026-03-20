import os
import json
import streamlit as st
from llama_cpp import Llama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ============================
# CONFIG
# ============================
MODEL_PATH = r"C:\Users\ahmad\Downloads\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

CHROMA_PATH = "chroma_db"
OUTPUT_JSON = "output1.json"

# ============================
# LOAD EMBEDDINGS
# ============================
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ============================
# LOAD LLM (LOCAL)
# ============================
@st.cache_resource
def load_llm():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=35,
        verbose=False
    )

llm = load_llm()
embedder = load_embedder()

# ============================
# LLM COMPLETION
# ============================
def get_completion(prompt: str) -> str:
    output = llm(
        prompt,
        max_tokens=512,
        stop=["</s>"]
    )
    # llama_cpp returns dict with 'choices' -> text
    text = output.get("choices", [{}])[0].get("text")
    if text is None:
        # fallback: some versions return 'content' or 'text' at top-level
        text = output.get("text", "") or str(output)
    return text.strip()

# ============================
# LOAD & INDEX JSON (معدّل لشكل الداتا اللي اعطيت)
# ============================
@st.cache_resource
def load_vector_db():
    if not os.path.exists(OUTPUT_JSON):
        st.error(f"❌ لم يتم العثور على {OUTPUT_JSON}")
        st.stop()

    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # نسمح بالصيغةين: (1) كائن فيه metadata + chunks  (2) أو قائمة من مستندات بسيطة
    docs = []
    if isinstance(data, dict) and "chunks" in data:
        global_meta = data.get("metadata", {})
        source_default = global_meta.get("source") or global_meta.get("source_file") or "unknown"
        language = global_meta.get("language", None)

        for chunk in data["chunks"]:
            text = chunk.get("text") or chunk.get("content") or ""
            chunk_id = chunk.get("id") or chunk.get("chunk_idx") or None
            # metadata القائم على كل قطعة
            md = {
                "source": chunk.get("source", source_default),
                "orig_source": source_default,
                "chunk_id": chunk_id,
                "word_count": chunk.get("word_count"),
                "char_count": chunk.get("char_count"),
                "language": chunk.get("language", language)
            }
            docs.append(
                Document(
                    page_content=text,
                    metadata=md
                )
            )
    elif isinstance(data, list):
        # حالة كان ملف JSON مصفوفة من مستندات بسيطة
        for item in data:
            text = item.get("content") or item.get("text") or ""
            md = {
                "source": item.get("source", "unknown"),
                "page": item.get("page", None),
                "language": item.get("language", None)
            }
            docs.append(Document(page_content=text, metadata=md))
    else:
        st.error("❌ شكل الـ JSON غير مدعوم. يجب أن يحتوي إما كائن به 'chunks' أو قائمة من المستندات.")
        st.stop()

    # لو حجم النصوص كبير، بنجزّئ كل وثيقة لقطع أصغر قبل الفهرسة
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(docs)

    # نضيف metadata لكل chunk لإبقاء مصدر القطعة واضح
    # Chroma.from_documents يتوقع Document objects (langchain schema) مع metadata
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_PATH
    )

    # persist لو الداتا كبيرة (بعض إصدارات Chroma يحتاج طريقة persist منفصلة)
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
        src = doc.metadata.get("source") or doc.metadata.get("orig_source") or "unknown"
        chunk_id = doc.metadata.get("chunk_id") or i
        lang = doc.metadata.get("language", "unknown")
        result_str += f"""
<SEARCH_RESULT>
<DOCUMENT>{doc.page_content}</DOCUMENT>
<SOURCE>{src}</SOURCE>
<CHUNK_ID>{chunk_id}</CHUNK_ID>
<LANG>{lang}</LANG>
</SEARCH_RESULT>
"""
    return result_str

# ============================
# RAG PROMPT
# ============================
def make_rag_prompt(query, result_str):
    return f"""
<INSTRUCTIONS>
أجب على السؤال باستخدام النتائج فقط.
إذا لم تجد الجواب في النتائج، قل "لا يوجد معلومات كافية".
اذكر المصدر في النهاية (اسم الملف والقطعة).
</INSTRUCTIONS>

<USER_QUERY>
{query}
</USER_QUERY>

<SEARCH_RESULTS>
{result_str}
</SEARCH_RESULTS>

ANSWER:
"""

# ============================
# CQR (إعادة صياغة سؤال سياقي)
# ============================
def rewrite_query(query, chat_history):
    prompt = f"""
أعد صياغة السؤال التالي اعتماداً على المحادثة السابقة:

<CHAT_HISTORY>
{chat_history}
</CHAT_HISTORY>

<QUESTION>
{query}
</QUESTION>

REWRITTEN:
"""
    return get_completion(prompt)

# ============================
# HyDE PROMPT
# ============================
def make_hyde_prompt(query):
    return f"""
أنشئ إجابة افتراضية قصيرة (فقرة واحدة) لهذا السؤال لاستخدامها في البحث:

<QUERY>
{query}
</QUERY>

HYPOTHETICAL ANSWER:
"""

# ============================
# HYDE + RAG
# ============================
def answer_with_hyde(query):
    hyde_prompt = make_hyde_prompt(query)
    hyde_answer = get_completion(hyde_prompt)

    result_str = populate_rag_query(hyde_answer)
    rag_prompt = make_rag_prompt(query, result_str)

    return get_completion(rag_prompt)

# ============================
# CQR + RAG
# ============================
def perform_cqr_rag(query, chat_history):
    refined_query = rewrite_query(query, chat_history)
    result_str = populate_rag_query(refined_query)
    rag_prompt = make_rag_prompt(refined_query, result_str)

    return refined_query, get_completion(rag_prompt)

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="Local RAG + HyDE + CQR", layout="wide")
st.title("🧠 Faqiah")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

user_query = st.text_input("اكتب سؤالك:")

mode = st.selectbox("اختر نمط المعالجة:", ["RAG ", "CQR + RAG", "HyDE + RAG"])

if st.button("إجابة"):
    if user_query.strip() == "":
        st.warning("❗ أدخل سؤالًا")
    else:
        if mode == "RAG ":
            result_str = populate_rag_query(user_query)
            prompt = make_rag_prompt(user_query, result_str)
            answer = get_completion(prompt)

        elif mode == "CQR + RAG":
            refined, answer = perform_cqr_rag(
                user_query,
                st.session_state.chat_history
            )
            st.info(f"🔁 السؤال بعد CQR: {refined}")

        elif mode == "HyDE + RAG":
            answer = answer_with_hyde(user_query)

        st.session_state.chat_history += f"user: {user_query}\nassistant: {answer}\n"
        st.success(answer)
