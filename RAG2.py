# -*- coding: utf-8 -*-
"""
Streamlit Debug + Mistral 7B Low Memory (GGUF)
- واجهة تفتح فورًا
- تحميل الموديل باستخدام llama-cpp-python
- توليد نصوص قصيرة للتجربة السريعة
"""

import os
import streamlit as st
from llama_cpp import Llama

# ============================
# Paths & Config
# ============================
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MISTRAL_PATH",
    "/home/albasha/models/mistral-7b-instruct-v0.2-gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

DEVICE = "cuda"  # llama-cpp-python يستعمل GPU تلقائيًا إذا CUDA متاحة

# ============================
# تحميل الموديل
# ============================
@st.cache_resource(show_spinner=True)
def load_local_mistral():
    try:
        llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=35,  # عدّل حسب VRAM عندك
            n_threads=8       # عدّل حسب عدد أنوية CPU
        )
        return llm
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
        return None

def generate_local_mistral(prompt: str, max_new_tokens: int = 50):
    llm = load_local_mistral()
    if llm is None:
        return "فشل تحميل الموديل."

    full_prompt = f"[INST]{prompt}[/INST]"

    try:
        output = llm(
            full_prompt,
            max_tokens=max_new_tokens,
            stop=["</s>"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        return f"حدث خطأ أثناء توليد النص: {e}"

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="🧠 Mistral Debug LowMem (GGUF)", page_icon="🧠")
st.title("🧠 Mistral 7B Local Debug (Low Memory) - GGUF")

st.markdown(f"**الجهاز:** {DEVICE}")
st.markdown("هذه نسخة Debug لتجربة توليد النصوص بدون استهلاك زائد للذاكرة.")

# Input
user_query = st.text_input("أدخل نصًا للتجربة:")

if st.button("توليد"):
    if user_query.strip():
        with st.spinner("يتم توليد النص..."):
            answer = generate_local_mistral(user_query)
            st.subheader("النتيجة")
            st.write(answer)
    else:
        st.warning("يرجى إدخال نص للتجربة.")
