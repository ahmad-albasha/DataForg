# -*- coding: utf-8 -*-
"""
Streamlit app (Arabic) with a single "إجابة" button that combines:
- CQR (Conversational Query Rewriting)
- HyDE (Hypothetical Document Embeddings)
- RAG (Retrieval-Augmented Generation via Chroma)

Notes:
- The UI and interactions are in Arabic.
- Answers are generated in Arabic.
- You need: OPENAI_API_KEY in environment, chromadb installed (and optional langchain).
"""
import os
import json
from typing import List, Dict, Any

import streamlit as st

# --- Optional dependencies ---
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ============================
# Configuration & Clients
# ============================
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _init_openai_client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI client غير متاح. ثبّت حزمة openai واضبط المتغيّر OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


def _init_chroma(persist_dir: str = ".chroma") -> Any:
    if chromadb is None:
        raise RuntimeError("ChromaDB غير متاح. رجاءً ثبّت chromadb.")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(name="documents")
    return client, collection


# ============================
# Utilities
# ============================
def _split_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    if RecursiveCharacterTextSplitter is None:
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            start = max(end - chunk_overlap, end)
        return chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def add_texts_to_collection(collection: Any, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
    if not texts:
        return
    collection.add(documents=texts, metadatas=metadatas, ids=ids)


def _preload_documents(collection: Any) -> None:
    """إضافة مستندات مدمجة مباشرة في الكود"""

    def _preload_documents(collection: Any) -> None:
        """إضافة مستندات من ملف txt مدمج في الكود"""
        try:
            # قراءة محتوى الملف (يمكنك تغيير المسار إذا كان الملف في مكان آخر)
            with open('New Text Document.txt', 'r', encoding='utf-8') as file:
                content = file.read()

            # تقسيم المحتوى إلى فقرات (افتراضاً أن كل فقرة مفصولة بسطرين جديدين)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            ids, texts, metas = [], [], []
            for i, para in enumerate(paragraphs):
                if not para:
                    continue

                # تقسيم كل فقرة إلى أجزاء (chunks) إذا كانت طويلة
                chunks = _split_text(para)

                for j, chunk in enumerate(chunks):
                    ids.append(f"doc-{i}-{j}")
                    texts.append(chunk)
                    metas.append({
                        "title": f"المستند {i + 1} - الجزء {j + 1}",
                        "author": "نظام المعرفة",
                        "doc_id": f"doc-{i}",
                        "chunk_idx": j,
                        "source_url": ""
                    })

            if texts:
                add_texts_to_collection(collection, texts, metas, ids)
                print(f"تم تحميل {len(texts)} جزء/أجزاء من الملف النصي.")

        except FileNotFoundError:
            print("ملف data.txt غير موجود، سيتم المتابعة بدون تحميل مستندات مسبقة.")
        except Exception as e:
            print(f"حدث خطأ أثناء قراءة الملف: {e}")


# ============================
# Retrieval + Prompts
# ============================
def populate_rag_query(query: str, n_results: int = 3) -> str:
    """
    ينفّذ بحثاً دلالياً على Chroma ويعيد سلسلة من كتل <SEARCH RESULT>.
    """
    _, collection = _init_chroma()
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
        <TITLE>{meta.get('title', 'N/A')}</TITLE>
        <AUTHOR>{meta.get('author', 'N/A')}</AUTHOR>
        <CHUNK_IDX>{meta.get('chunk_idx', 'N/A')}</CHUNK_IDX>
        <DOC_ID>{meta.get('doc_id', 'N/A')}</DOC_ID>
        <URL>{meta.get('source_url', 'N/A')}</URL>
    </METADATA>
</SEARCH RESULT>"""
        )
    return "\\n".join(formatted)


def make_decoupled_rag_prompt_ar(user_query: str, results: str) -> str:
    """
    حافز "مفصول" بالعربية: استخراج حقائق ثم الإجابة بالرجوع للمصادر.
    """
    return f"""أنت مساعد عربي دقيق.

[المهمة 1: استخراج الحقائق]
استخرج حقائق ذرّية ومحدّدة من <RESULTS> ذات صلة بالسؤال.
أدرجها تحت قسم بعنوان: الحقائق.

[المهمة 2: الإجابة]
استخدم "الحقائق" فقط للإجابة عن السؤال بالعربية الفصحى الواضحة.
عند الاقتباس من نتيجة، اذكر رابط المصدر (URL) الموجود في النتيجة.
إذا كانت المعلومات غير كافية، قل بصراحة "لا أعلم من المصادر المتاحة".

<السؤال>
{user_query}
</السؤال>

<RESULTS>
{results}
</RESULTS>
"""


def make_hyde_prompt_ar(user_query: str) -> str:
    return f"""اكتب إجابة عربية موجزة وواقعية ومحتملة حول السؤال التالي.
هذه الإجابة افتراضية فقط لتحسين الاسترجاع وليست الإجابة النهائية.

السؤال: {user_query}
الإجابة الافتراضية:"""


def make_cqr_prompt_ar(user_query: str) -> str:
    return f"""أعد كتابة الاستعلام التالي إلى سؤال عربي واضح ومكتفٍ ذاتياً،
مع المحافظة على المقصود الأصلي دون إضافة معلومات من عندك.

الاستعلام الأصلي: {user_query}
الصياغة المنقّحة:"""


def get_completion(prompt: str, model: str = OPENAI_MODEL_DEFAULT) -> str:
    """
    يستدعي OpenAI مع رسالة نظام لضمان العربية + الدقّة.
    """
    client = _init_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "أجب دائماً باللغة العربية الفصحى وبأسلوب واضح ومختصر. عندما تقتبس من نتائج RAG اذكر الروابط."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def combined_answer(user_query: str, n_results: int = 3, model: str = OPENAI_MODEL_DEFAULT) -> str:
    """
    المسار الموحَّد (زر واحد):
      1) CQR: إعادة صياغة الاستعلام.
      2) HyDE: توليد إجابة افتراضية للاسترجاع.
      3) RAG: استرجاع باستخدام نص HyDE ثم الإجابة باستخدام حافز مفصول.
    """
    # 1) CQR
    rewritten = get_completion(make_cqr_prompt_ar(user_query), model=model).strip()
    if not rewritten:
        rewritten = user_query

    # 2) HyDE بناءً على الصياغة المنقّحة
    hyde_seed = get_completion(make_hyde_prompt_ar(rewritten), model=model)

    # 3) استرجاع + إجابة
    results = populate_rag_query(hyde_seed, n_results=n_results)
    prompt = make_decoupled_rag_prompt_ar(user_query, results)
    return get_completion(prompt, model=model)


# ============================
# Streamlit UI (Arabic)
# ============================
def _rtl_css():
    st.markdown(
        """
        <style>
        body, .stMarkdown, .stTextInput, .stButton, .stSlider, .stFileUploader, .stExpander, .stAlert {
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
    st.set_page_config(page_title="📚 صندوق إجابة واحد (CQR + HyDE + RAG)", page_icon="🧠")
    _rtl_css()

    st.title("🧠TextWise")
    st.caption(": Chroma: {} | OpenAI: {}".format("✅" if chromadb is not None else "❌",
                                                  "✅" if (OpenAI is not None and os.getenv("OPENAI_API_KEY")) else "❌"))

    # تهيئة Chroma وإضافة المستندات المدمجة
    try:
        client, collection = _init_chroma()
        _preload_documents(collection)
    except Exception as e:
        st.error(f"تعذّرت تهيئة Chroma: {e}")
        collection = None

    st.subheader("❓ اسأل سؤالك")
    user_query = st.text_input("اكتب سؤالك هنا بالعربية...")

    # عناصر تحكم غير تفاعلية (لا أزرار متعددة)
    n_results = st.slider("عدد النتائج المسترجعة", 1, 10, 3)

    # زر واحد فقط
    if st.button("إجابة"):
        if not user_query.strip():
            st.warning("يرجى كتابة سؤال أولاً.")
        else:
            try:
                answer = combined_answer(user_query, n_results=n_results)
                st.subheader("النتيجة")
                st.write(answer)
            except Exception as e:
                st.error(f"حدث خطأ أثناء توليد الإجابة: {e}")


if __name__ == "__main__":
    main()