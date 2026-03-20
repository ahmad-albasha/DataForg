# app.py

import os
import json
import streamlit as st
from llama_cpp import Llama

# ============================
# بدائل للوظائف المطلوبة
# ============================

# بديل لـ HuggingFaceEmbeddings
try:
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError:
    st.error("❌ لم يتم العثور على langchain.embeddings. جارٍ استخدام بديل...")
    try:
        # استخدام sentence-transformers مباشرة
        from sentence_transformers import SentenceTransformer
        import numpy as np


        class SimpleEmbeddings:
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)

            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()

            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()


        HuggingFaceEmbeddings = SimpleEmbeddings
    except ImportError:
        st.error("""
        ❌ يرجى تثبيت المكتبات المطلوبة:
        ```bash
        pip install sentence-transformers
        pip install langchain
        pip install chromadb
        pip install pypdf
        pip install docx2txt
        ```
        """)
        st.stop()

# بديل لـ Chroma
try:
    from langchain.vectorstores import Chroma
except ImportError:
    st.error("❌ لم يتم العثور على langchain.vectorstores. جارٍ التثبيت...")
    st.info("جارٍ تثبيت chromadb... قد يستغرق بضع دقائق")
    import subprocess
    import sys

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb", "langchain"])
        from langchain.vectorstores import Chroma

        st.success("✅ تم تثبيت المكتبات بنجاح")
        st.rerun()
    except:
        st.error("❌ فشل تثبيت المكتبات. يرجى التثبيت يدوياً.")
        st.stop()

# بديل لـ RecursiveCharacterTextSplitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    st.error("❌ لم يتم العثور على langchain.text_splitter")
    st.stop()

# بديل لـ loaders
try:
    from langchain.document_loaders import (
        TextLoader,
        PyPDFLoader,
        Docx2txtLoader,
        CSVLoader
    )
except ImportError:
    st.error("❌ لم يتم العثور على langchain.document_loaders")
    st.stop()

# ============================
# Settings
# ============================
MODEL_PATH = "/home/albasha/models/mistral-7b-instruct-v0.2-gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CHROMA_PATH = "chroma_db"
OUTPUT_JSON = "output.json"


# ============================
# Load LLM
# ============================
@st.cache_resource
def load_llm():
    try:
        return Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False
        )
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        st.stop()


# ============================
# Load Embeddings
# ============================
@st.cache_resource
def load_embedder():
    try:
        # محاولة استخدام HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embedder
    except Exception as e:
        st.error(f"❌ خطأ في تحميل نموذج التضمين: {e}")

        # استخدام sentence-transformers مباشرة كبديل
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            class SimpleEmbedder:
                def __init__(self):
                    self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

                def embed_documents(self, texts):
                    embeddings = self.model.encode(texts, convert_to_tensor=False)
                    return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

                def embed_query(self, text):
                    embedding = self.model.encode([text], convert_to_tensor=False)
                    return embedding[0].tolist() if hasattr(embedding[0], 'tolist') else embedding[0]

            st.info("✅ استخدام نموذج التضمين المبسط")
            return SimpleEmbedder()
        except Exception as e2:
            st.error(f"❌ فشل تحميل نموذج التضمين البديل: {e2}")
            st.stop()


# ============================
# Build or Load ChromaDB
# ============================
@st.cache_resource
def initialize_vector_db():
    """
    تهيئة قاعدة بيانات المتجهات من ملف output.json أو تحميلها إذا كانت موجودة مسبقاً
    """
    embedder = load_embedder()

    # التحقق إذا كانت قاعدة البيانات موجودة مسبقاً
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        st.info("📂 جارٍ تحميل قاعدة البيانات الموجودة...")
        try:
            vectordb = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedder
            )
            st.success("✅ تم تحميل قاعدة البيانات بنجاح")
            return vectordb
        except Exception as e:
            st.warning(f"⚠️ خطأ في تحميل قاعدة البيانات: {e}")
            st.info("🔨 سيتم إنشاء قاعدة بيانات جديدة...")

    # التحقق من وجود ملف output.json
    if not os.path.exists(OUTPUT_JSON):
        st.error("""
        ❌ ملف output.json غير موجود!

        **الرجاء اتباع الخطوات التالية:**

        1. **أضف ملفاتك إلى المجلد الحالي:**
           - ملفات نصية (.txt)
           - ملفات PDF (.pdf)
           - ملفات Word (.docx)
           - ملفات CSV (.csv)

        2. **قم بإنشاء ملف output.json يدوياً:**
           ```bash
           echo '["file1.txt", "file2.pdf"]' > output.json
           ```

        3. **أو استخدم المحرر المدمج أدناه لإنشاء الملف:**
        """)

        # محرر مدمج لإنشاء output.json
        st.subheader("✏️ إنشاء ملف output.json")

        # عرض الملفات الموجودة
        current_files = [f for f in os.listdir('.') if os.path.isfile(f)]
        if current_files:
            st.write("**الملفات الموجودة في المجلد:**")
            for file in current_files:
                st.write(f"- {file}")

        # محرر JSON
        default_content = json.dumps([f for f in current_files if f.endswith(('.txt', '.pdf', '.docx', '.csv'))],
                                     ensure_ascii=False, indent=2)
        json_content = st.text_area("محتوى output.json:", value=default_content, height=200)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 حفظ output.json"):
                try:
                    parsed = json.loads(json_content)
                    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)
                    st.success(f"✅ تم حفظ {OUTPUT_JSON}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ خطأ في حفظ الملف: {e}")

        with col2:
            if st.button("📁 تحميل من المجلد الحالي"):
                # إنشاء قائمة بجميع الملفات المدعومة
                supported_files = [
                    f for f in current_files
                    if f.endswith(('.txt', '.pdf', '.docx', '.csv'))
                ]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(supported_files, f, ensure_ascii=False, indent=2)
                st.success(f"✅ تم إنشاء {OUTPUT_JSON} مع {len(supported_files)} ملف")
                st.rerun()

        st.stop()

    # تحميل قائمة الملفات من output.json
    try:
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            files_list = json.load(f)
    except Exception as e:
        st.error(f"❌ خطأ في قراءة {OUTPUT_JSON}: {e}")
        st.stop()

    if not files_list:
        st.error("⚠️ ملف output.json فارغ! الرجاء إضافة مسارات الملفات.")
        st.stop()

    # تحميل المستندات من الملفات المحددة
    st.info(f"🔨 جارٍ إنشاء قاعدة بيانات من {len(files_list)} ملف...")
    docs = []
    successful_files = []
    failed_files = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file_path in enumerate(files_list):
        try:
            status_text.text(f"📖 جارٍ تحميل: {os.path.basename(file_path)}...")

            # التحقق من وجود الملف
            if not os.path.exists(file_path):
                st.warning(f"⚠️ الملف غير موجود: {file_path}")
                failed_files.append(file_path)
                continue

            # تحميل الملف حسب نوعه
            if file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith(".csv"):
                loader = CSVLoader(file_path, encoding="utf-8")
            else:
                st.warning(f"⚠️ امتداد غير مدعوم: {file_path}")
                failed_files.append(file_path)
                continue

            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            successful_files.append(file_path)

            progress_bar.progress((i + 1) / len(files_list))

        except Exception as e:
            st.warning(f"⚠️ خطأ في تحميل {file_path}: {e}")
            failed_files.append(file_path)
            continue

    status_text.empty()
    progress_bar.empty()

    # عرض نتائج التحميل
    if successful_files:
        st.success(f"✅ {len(successful_files)} ملف تم تحميله بنجاح")

    if failed_files:
        with st.expander("📋 الملفات التي فشل تحميلها", expanded=False):
            for file in failed_files:
                st.write(f"- {file}")

    if not docs:
        st.error("❌ لم يتم تحميل أي وثائق. لا يمكن إنشاء قاعدة البيانات.")
        st.stop()

    # تقسيم المستندات
    st.info("✂️ جارٍ تقسيم الوثائق...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=70,
        length_function=len
    )
    split_docs = splitter.split_documents(docs)

    st.success(f"📊 تم تقسيم الوثائق إلى {len(split_docs)} جزء")

    # إنشاء قاعدة بيانات المتجهات
    try:
        st.info("🏗️ جارٍ إنشاء قاعدة البيانات...")
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embedder,
            persist_directory=CHROMA_PATH
        )

        st.success(f"✅ تم إنشاء قاعدة البيانات في: {CHROMA_PATH}")

        # حفظ معلومات عن الملفات التي تم تحميلها بنجاح
        info_path = os.path.join(CHROMA_PATH, "loaded_files.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_chunks": len(split_docs),
                "creation_time": st.session_state.get('creation_time', 'unknown')
            }, f, ensure_ascii=False, indent=2)

        return vectordb

    except Exception as e:
        st.error(f"❌ خطأ في إنشاء قاعدة البيانات: {e}")
        st.stop()


# ============================
# RAG Search
# ============================
def get_context(query, k=3):
    try:
        vectordb = initialize_vector_db()
        results = vectordb.similarity_search(query, k=k)

        if not results:
            return ""

        context_parts = []
        for i, doc in enumerate(results):
            # استخراج اسم الملف من metadata إذا كان موجوداً
            source_name = "غير معروف"
            if hasattr(doc, 'metadata'):
                source = doc.metadata.get('source', '')
                if source:
                    source_name = os.path.basename(source)

            context_parts.append(f"📄 **المصدر {i + 1}** ({source_name}):\n{doc.page_content}\n")

        return "\n".join(context_parts)
    except Exception as e:
        st.error(f"❌ خطأ في البحث: {e}")
        return ""


# ============================
# Generate Answer
# ============================
def generate_answer(query):
    try:
        context = get_context(query)
        if not context:
            return "❌ لا يمكن العثور على معلومات في المصادر المتاحة. الرجاء التحقق من أن الملفات تحتوي على المعلومات المطلوبة."

        llm = load_llm()

        prompt = f"""<s>[INST] أنا مساعد ذكي. سأجيب على أسئلتك بناءً على المصادر المتاحة فقط.

المصادر المتاحة:
{context}

بناءً على المصادر المذكورة أعلاه فقط، أجب على السؤال التالي:
السؤال: {query}

ملاحظات مهمة:
1. إذا لم أجد الجواب في المصادر، سأقول: "غير موجود في المصادر المتاحة."
2. سأكون دقيقاً وأشير إلى المعلومات من المصادر.
3. سأكتب الإجابة باللغة العربية الفصحى.
4. سأقوم بتحليل المعلومات وتلخيصها بشكل واضح.

الإجابة: [/INST]"""

        response = llm(
            prompt,
            max_tokens=512,
            temperature=0.1,
            top_p=0.9,
            stop=["</s>", "[INST]", "[/INST]"],
            echo=False
        )

        answer = response["choices"][0]["text"].strip()
        return answer

    except Exception as e:
        return f"❌ حدث خطأ أثناء توليد الإجابة: {e}"


# ============================
# إدارة الملفات
# ============================
def manage_output_json():
    """
    واجهة لإدارة ملف output.json
    """
    st.sidebar.subheader("📁 إدارة الملفات")

    # عرض الملفات الحالية
    current_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    supported_files = [f for f in current_files if f.endswith(('.txt', '.pdf', '.docx', '.csv'))]

    if st.sidebar.button("🔄 تحديث قائمة الملفات", key="refresh_files"):
        if supported_files:
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(supported_files, f, ensure_ascii=False, indent=2)
            st.sidebar.success(f"✅ تم تحديث {OUTPUT_JSON}")
            st.rerun()
        else:
            st.sidebar.warning("⚠️ لا توجد ملفات مدعومة")

    if os.path.exists(OUTPUT_JSON):
        if st.sidebar.button("👁️ عرض output.json", key="view_output"):
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                content = json.load(f)
            st.sidebar.write("**محتوى output.json:**")
            st.sidebar.json(content)

    if st.sidebar.button("🗑️ إعادة إنشاء قاعدة البيانات", type="secondary", key="rebuild_db"):
        if os.path.exists(CHROMA_PATH):
            import shutil
            shutil.rmtree(CHROMA_PATH)
            st.cache_resource.clear()
            st.sidebar.success("✅ تم مسح قاعدة البيانات القديمة")
            st.rerun()


# ============================
# Streamlit UI
# ============================
st.set_page_config(
    page_title="نظام RAG مع Mistral",
    page_icon="📚",
    layout="wide"
)

# إضافة وقت الإنشاء إلى session state
if 'creation_time' not in st.session_state:
    from datetime import datetime

    st.session_state.creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.title("📚 نظام البحث والإجابة الذكي (RAG)")
st.markdown("---")

# إدارة الملفات في الشريط الجانبي
manage_output_json()

# قسم المعلومات
with st.expander("🚀 بدء الاستخدام", expanded=True):
    st.markdown("""
    ### 📋 **خطوات التشغيل:**

    1. **وضع الملفات** في المجلد الحالي
    2. **تحديث قائمة الملفات** من الشريط الجانبي
    3. **كتابة السؤال** في المربع أدناه
    4. **النقر على زر البحث** للحصول على الإجابة

    ### 📄 **الملفات المدعومة:**
    - 📝 ملفات نصية (.txt)
    - 📄 ملفات PDF (.pdf)
    - 📘 ملفات Word (.docx)
    - 📊 ملفات CSV (.csv)

    ### ⚙️ **تثبيت المتطلبات:**
    ```bash
    pip install streamlit llama-cpp-python
    pip install langchain chromadb
    pip install sentence-transformers
    pip install pypdf docx2txt
    ```
    """)

# قسم الاستعلام
st.subheader("💬 اسأل عن محتوى مستنداتك")

query = st.text_area(
    "اكتب سؤالك هنا:",
    placeholder="مثال: ما هي أهم النقاط في الوثيقة؟\nأو: ما هو تعريف المصطلح X؟",
    height=100,
    key="query_input"
)

col1, col2 = st.columns([3, 1])
with col2:
    search_button = st.button("🔍 بحث وتوليد الإجابة", type="primary", use_container_width=True)

# قسم النتائج
if search_button and query.strip():
    with st.spinner("🔍 جارٍ البحث في المصادر..."):
        context = get_context(query)

    if context:
        with st.expander("📄 عرض المصادر المسترجعة", expanded=True):
            st.markdown(context)
    else:
        st.warning("⚠️ لم يتم العثور على معلومات ذات صلة في المصادر.")

    with st.spinner("🤖 جارٍ توليد الإجابة... قد يستغرق بضع ثوان"):
        answer = generate_answer(query)

        st.markdown("---")
        st.subheader("📝 النتيجة")

        st.markdown(f"**السؤال:** `{query}`")
        st.markdown("**الإجابة:**")

        # عرض الإجابة في مربع مخصص
        st.info(answer)

        # أزرار تفاعلية
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 سؤال جديد", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("📋 نسخ الإجابة", use_container_width=True):
                st.success("تم نسخ الإجابة إلى الحافظة!")

elif search_button and not query.strip():
    st.warning("⚠️ الرجاء كتابة سؤال أولاً!")

# قسم إحصاءات قاعدة البيانات
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 إحصاءات النظام")

    # معلومات قاعدة البيانات
    if os.path.exists(CHROMA_PATH):
        try:
            info_file = os.path.join(CHROMA_PATH, "loaded_files.json")
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    db_info = json.load(f)

                st.metric("📁 الملفات المحملة", len(db_info.get('successful_files', [])))
                st.metric("✂️ عدد الأجزاء", db_info.get('total_chunks', 0))
            else:
                db_files = os.listdir(CHROMA_PATH)
                st.write(f"🗃️ قاعدة بيانات موجودة ({len(db_files)} ملف)")
        except:
            st.write("📊 قاعدة البيانات جاهزة")
    else:
        st.info("🗄️ قاعدة البيانات غير موجودة بعد")

    # معلومات output.json
    st.markdown("---")
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                files = json.load(f)

            st.write(f"📋 **الملفات المحددة:** {len(files)}")

            # عد أنواع الملفات
            from collections import Counter

            extensions = Counter([os.path.splitext(f)[1].lower() for f in files])

            for ext, count in extensions.items():
                if ext:
                    st.write(f"{ext}: {count}")
        except:
            st.write("📄 ملف output.json موجود")

# تذييل الصفحة
st.markdown("---")
st.caption(f"✨ نظام RAG الذكي | تم الإنشاء في: {st.session_state.creation_time}")

# ============================
# نصائح تثبيت المكتبات
# ============================
if st.sidebar.button("🛠️ تثبيت المكتبات المطلوبة"):
    st.sidebar.info("""
    **أوامر التثبيت:**

    ```bash
    # المكتبات الأساسية
    pip install streamlit llama-cpp-python

    # معالجة النصوص والبحث
    pip install langchain chromadb

    # التضمين والنماذج
    pip install sentence-transformers

    # معالجة الملفات
    pip install pypdf docx2txt

    # تحديث pip
    pip install --upgrade pip
    ```

    **للتحقق من التثبيت:**
    ```bash
    python -c "import streamlit, langchain, chromadb; print('✅ جميع المكتبات مثبتة')"
    ```
    """)