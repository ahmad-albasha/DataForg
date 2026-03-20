# -*- coding: utf-8 -*-
"""
Streamlit RAG app (Arabic) using:
- Local Mistral model from HuggingFace (via transformers)
- Local JSON datasets with pre-chunked text
- ChromaDB for retrieval

Requirements:
  pip install streamlit transformers accelerate torch chromadb

Assumptions:
- Your local model is in LOCAL_MODEL_PATH
- Your datasets are JSON files in ./datasets with structure:
  {
    "metadata": {...},
    "chunks": [
      {
        "id": 1,
        "text": "...",
        "word_count": ...,
        "char_count": ...
      },
      ...
    ]
  }
"""

import os
import json
from typing import List, Dict, Any

import streamlit as st
import chromadb
from chromadb.config import Settings

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ============================
# Paths & Config
# ============================

# عدّل هذا المسار حسب مكان الموديل عندك
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MISTRAL_PATH",
    "/home/albasha/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
)

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), ".chroma_local")

# جهاز التنفيذ: GPU إذا متوفر، غير ذلك CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ============================
# Model Loading (Mistral Local)
# ============================

@st.cache_resource(show_spinner=True)
@st.cache_resource(show_spinner=True)
def load_local_mistral():
    """
    تحميل موديل Mistral من المسار المحلي باستخدام transformers.
    """
    if not os.path.isdir(LOCAL_MODEL_PATH):
        raise RuntimeError(f"مسار الموديل غير موجود: {LOCAL_MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=DTYPE,
    )

    model.to(DEVICE)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model



def generate_local_mistral(prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    دالة مساعدة لاستدعاء الموديل المحلي وتوليد نص عربي بناءً على prompt.
    """
    tokenizer, model = load_local_mistral()

    # نمط محادثة بسيط (System + User)
    full_prompt = f"""[INST]أجب دائماً باللغة العربية الفصحى وبأسلوب واضح ومختصر.

{prompt}[/INST]
"""

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    # نفصل النص المولّد بعد الـ prompt
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    # نزيل الـ prompt من البداية إن وجد
    if full_prompt in generated:
        generated = generated.replace(full_prompt, "").strip()
    return generated.strip()


# ============================
# ChromaDB
# ============================

@st.cache_resource(show_spinner=True)
def init_chroma() -> chromadb.Client:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(allow_reset=True)
    )
    return client


def get_collection(client: chromadb.Client, name: str):
    return client.get_or_create_collection(name=name)


def add_chunks_to_collection(collection: Any, chunks: List[Dict[str, Any]]):
    """
    chunks: قائمة من عناصر فيها keys: id, text, meta
    """
    if not chunks:
        return
    ids = [str(c["id"]) for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c.get("meta", {}) for c in chunks]
    collection.add(ids=ids, documents=texts, metadatas=metadatas)


# ============================
# Dataset Loading
# ============================

def load_dataset_names() -> List[str]:
    """
    ترجع قائمة بأسماء الـ datasets بدون .json من مجلد datasets.
    """
    if not os.path.isdir(DATASETS_DIR):
        return []
    names = []
    for fname in os.listdir(DATASETS_DIR):
        if fname.endswith(".json"):
            names.append(os.path.splitext(fname)[0])
    return sorted(names)


def load_dataset_chunks(name: str) -> List[Dict[str, Any]]:
    """
    يقرأ ملف JSON من مجلد datasets بالشكل:
    {
      "metadata": {...},
      "chunks": [
        { "id": 1, "text": "...", "word_count": ..., "char_count": ... },
        ...
      ]
    }

    ويرجّع قائمة chunks بالشكل الموحد:
    [
      {
        "id": "fia1-1",
        "text": "...",
        "meta": {
          "source": "fia1.pdf",
          "doc_id": "fia1",
          "chunk_idx": 1,
          "word_count": ...,
          "char_count": ...
        }
      },
      ...
    ]
    """
    path = os.path.join(DATASETS_DIR, name + ".json")
    if not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # نتوقع أنه dict فيه "metadata" و "chunks"
    if not isinstance(data, dict) or "chunks" not in data:
        return []

    meta_global = data.get("metadata", {})
    chunks_raw = data.get("chunks", [])

    chunks: List[Dict[str, Any]] = []
    for ch in chunks_raw:
        # ch متوقّع يكون dict فيه id, text, word_count, char_count
        ch_id = ch.get("id")
        txt = ch.get("text", "")
        word_count = ch.get("word_count")
        char_count = ch.get("char_count")

        # نخلي id نصي + نضيف اسم الملف
        final_id = f"{name}-{ch_id}"

        chunks.append({
            "id": str(final_id),
            "text": txt,
            "meta": {
                "source": meta_global.get("source", name + ".pdf"),
                "doc_id": name,
                "chunk_idx": ch_id,
                "word_count": word_count,
                "char_count": char_count,
                "chunk_size": meta_global.get("chunk_size"),
                "overlap": meta_global.get("overlap"),
                "language": meta_global.get("language"),
            },
        })

    return chunks


# ============================
# RAG Helpers
# ============================

def populate_rag_query(client: chromadb.Client, dataset: str, query: str, n_results: int = 3) -> str:
    """
    ينفّذ بحثاً دلالياً على مجموعة بيانات معيّنة في Chroma ويعيد كتل <SEARCH RESULT>.
    """
    collection_name = f"ds_{dataset}"
    collection = get_collection(client, collection_name)

    results = collection.query(query_texts=[query], n_results=n_results)
    if not results.get("documents"):
        return ""

    docs = results["documents"][0]
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else [{} for _ in docs]

    formatted = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        formatted.append(
            f"""<SEARCH RESULT>
<DOCUMENT>{doc}</DOCUMENT>
<METADATA>
    <TITLE>{meta.get('title', meta.get('source', 'N/A'))}</TITLE>
    <DOC_ID>{meta.get('doc_id', 'N/A')}</DOC_ID>
    <CHUNK_IDX>{meta.get('chunk_idx', 'N/A')}</CHUNK_IDX>
</METADATA>
</SEARCH RESULT>"""
        )
    return "\n".join(formatted)


def make_decoupled_rag_prompt_ar(user_query: str, results: str) -> str:
    """
    حافز مفصول: استخراج حقائق ثم الإجابة.
    """
    return f"""أنت مساعد عربي دقيق.

[المهمة 1: استخراج الحقائق]
استخرج حقائق ذرّية ومحدّدة من <RESULTS> ذات صلة بالسؤال.
أدرجها تحت قسم بعنوان: الحقائق.

[المهمة 2: الإجابة]
استخدم "الحقائق" فقط للإجابة عن السؤال بالعربية الفصحى الواضحة.
إذا كانت المعلومات غير كافية، قل بصراحة "لا أعلم من المصادر المتاحة".

<السؤال>
{user_query}
</السؤال>

<RESULTS>
{results}
</RESULTS>
"""


def make_cqr_prompt_ar(user_query: str) -> str:
    return f"""أعد كتابة الاستعلام التالي إلى سؤال عربي واضح ومكتفٍ ذاتياً،
مع المحافظة على المقصود الأصلي دون إضافة معلومات من عندك.

الاستعلام الأصلي: {user_query}
الصياغة المنقّحة:"""


def make_hyde_prompt_ar(user_query: str) -> str:
    return f"""اكتب إجابة عربية موجزة وواقعية ومحتملة حول السؤال التالي.
هذه الإجابة افتراضية فقط لتحسين الاسترجاع وليست الإجابة النهائية.

السؤال: {user_query}
الإجابة الافتراضية:"""


def combined_answer_local(
    client: chromadb.Client,
    dataset: str,
    user_query: str,
    n_results: int = 3,
) -> str:
    """
    المسار الموحّد (محلي):
      1) CQR: إعادة صياغة الاستعلام باستخدام موديل Mistral المحلي.
      2) HyDE: توليد إجابة افتراضية للاسترجاع.
      3) RAG: استرجاع باستخدام نص HyDE ثم الإجابة باستخدام حافز مفصول.
    """
    # 1) CQR
    rewritten = generate_local_mistral(make_cqr_prompt_ar(user_query), max_new_tokens=128).strip()
    if not rewritten:
        rewritten = user_query

    # 2) HyDE
    hyde_seed = generate_local_mistral(make_hyde_prompt_ar(rewritten), max_new_tokens=256)

    # 3) Retrieval + Answer
    results = populate_rag_query(client, dataset, hyde_seed, n_results=n_results)
    prompt = make_decoupled_rag_prompt_ar(user_query, results)
    answer = generate_local_mistral(prompt, max_new_tokens=512)
    return answer


# ============================
# Streamlit UI
# ============================

def _rtl_css():
    st.markdown(
        """
        <style>
        body, .stMarkdown, .stTextInput, .stButton, .stSlider, .stSelectbox, .stRadio {
            direction: rtl;
            text-align: right;
            font-family: "Tahoma", "Cairo", sans-serif;
        }
        .block-container { padding-top: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="📚 RAG محلي مع Mistral", page_icon="🧠")
    _rtl_css()

    st.title("🧠 TextWise - إصدار محلي (Mistral + RAG)")
    st.caption(f"الجهاز: {DEVICE} | الموديل: {os.path.basename(LOCAL_MODEL_PATH)}")

    # تحميل الموديل (مرة واحدة)
    with st.spinner("يتم تحميل الموديل المحلي..."):
        _ = load_local_mistral()

    # تهيئة Chroma
    client = init_chroma()

    # اختيار مجموعة البيانات
    datasets = load_dataset_names()
    if not datasets:
        st.error("لا يوجد أي ملفات JSON في مجلد datasets. تأكد من وجود ملفات هناك.")
        return

    selected_dataset = st.selectbox("اختر مجموعة البيانات التي تريد استخدامها:", datasets)

    # زر تحميل البيانات إلى Chroma (مرة لكل dataset)
    if st.button("⚙️ تجهيز قاعدة المعرفة (index) لهذه البيانات"):
        chunks = load_dataset_chunks(selected_dataset)
        if not chunks:
            st.error(f"ملف dataset `{selected_dataset}.json` فارغ أو غير صالح.")
        else:
            collection_name = f"ds_{selected_dataset}"
            collection = get_collection(client, collection_name)
            # ممكن نعيد ملأها في كل مرة (أو تضيف من دون reset حسب رغبتك)
            collection.delete(where={})  # مسح قديم
            add_chunks_to_collection(collection, chunks)
            st.success(f"تم تحميل {len(chunks)} جزء/أجزاء إلى مجموعة: {collection_name}")

    st.subheader("❓ اسأل سؤالك")
    user_query = st.text_input("اكتب سؤالك هنا بالعربية...")

    n_results = st.slider("عدد النتائج المسترجعة", 1, 10, 3)

    if st.button("إجابة"):
        if not user_query.strip():
            st.warning("يرجى كتابة سؤال أولاً.")
        else:
            with st.spinner("يتم توليد الإجابة باستخدام الموديل المحلي..."):
                try:
                    answer = combined_answer_local(
                        client=client,
                        dataset=selected_dataset,
                        user_query=user_query,
                        n_results=n_results,
                    )
                    st.subheader("النتيجة")
                    st.write(answer)
                except Exception as e:
                    st.error(f"حدث خطأ أثناء توليد الإجابة: {e}")


if __name__ == "__main__":
    main()
