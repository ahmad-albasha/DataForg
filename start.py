

import fitz


print(fitz.__doc__)

# =========================
# PDF PROCESSOR
# =========================
import sys
import os
import json
import re
import subprocess
import tempfile
import io

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import fitz
import pytesseract
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ArabicPDFProcessor:
    def __init__(self):
        self.reshaping_config = {
            'language': 'ar',
            'support_ligatures': True,
            'delete_harakat': True
        }
        self.poppler_path = self._find_poppler()

    def _find_poppler(self):
        possible_paths = [
            r'C:\poppler\poppler-25.12.0\Library\bin',
            r'C:\Program Files\poppler\bin',
            r'C:\Program Files (x86)\poppler\bin',
            '/usr/bin',
            '/usr/local/bin',
            '/opt/homebrew/bin',
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"تم العثور على Poppler في: {path}")
                return path

        print("لم يتم العثور على Poppler، جاري استخدام بدائل...")
        return None

    def extract_text_from_pdf(self, pdf_path):
        text = ""

        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"

            if not text.strip():
                print("لم يتم العثور على نص، جاري معالجة الصور...")
                text = self._extract_text_from_scanned_pdf(pdf_path)
            else:
                print("تم استخراج النص مباشرة من PDF")

        except Exception as e:
            print(f"خطأ في معالجة PDF: {e}")
            text = self._extract_text_from_scanned_pdf(pdf_path)

        return text

    def _extract_text_from_scanned_pdf(self, pdf_path):
        text = ""

        try:
            text = self._extract_text_with_pymupdf(pdf_path)

            if not text.strip():
                text = self._extract_text_with_pdf2image(pdf_path)

        except Exception as e:
            print(f"خطأ في معالجة PDF الممسوح: {e}")

        return text

    def _extract_text_with_pymupdf(self, pdf_path):
        text = ""
        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                    continue

                image_list = page.get_images()
                if image_list:
                    print(f"معالجة {len(image_list)} صورة في الصفحة {page_num + 1}...")

                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]

                        image = Image.open(io.BytesIO(image_bytes))

                        try:
                            img_text = pytesseract.image_to_string(image, lang='ara')
                            text += img_text + "\n"
                        except:
                            try:
                                img_text = pytesseract.image_to_string(image, lang='eng')
                                text += img_text + "\n"
                            except Exception as e:
                                print(f"خطأ في OCR للصورة {img_index + 1}: {e}")

            doc.close()

        except Exception as e:
            print(f"خطأ في استخراج النص باستخدام PyMuPDF: {e}")

        return text

    def _extract_text_with_pdf2image(self, pdf_path):
        text = ""

        try:
            try:
                from pdf2image import convert_from_path
            except ImportError:
                print("pdf2image غير مثبت، جاري تثبيته...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pdf2image'])
                from pdf2image import convert_from_path

            images = convert_from_path(
                pdf_path,
                dpi=200,
                poppler_path=self.poppler_path
            )

            for i, image in enumerate(images):
                print(f"معالجة الصفحة {i + 1} من {len(images)}...")

                try:
                    page_text = pytesseract.image_to_string(image, lang='ara')
                    text += page_text + "\n"
                except:
                    try:
                        page_text = pytesseract.image_to_string(image, lang='eng')
                        text += page_text + "\n"
                    except Exception as e:
                        print(f"فشل OCR للصفحة {i + 1}: {e}")

        except Exception as e:
            print(f"خطأ في استخدام pdf2image: {e}")

        return text

    def clean_arabic_text(self, text):
        if not text or not text.strip():
            return ""

        text = re.sub(
            r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\.\,\!\?\:\;\(\)\[\]\{\}0-9a-zA-Z]',
            '',
            text
        )
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if not text:
            return ""

        try:
            text = arabic_reshaper.reshape(text, self.reshaping_config)
            text = get_display(text)
        except Exception as e:
            print(f"خطأ في تحسين النص العربي: {e}")

        return text

    def split_into_chunks(self, text, chunk_size=300, overlap=30):
        if not text or not text.strip():
            return []

        sentences = re.split(r'[.!?؟۔]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            words = sentence.split()
            sentence_size = len(words)

            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

                if chunks and overlap > 0:
                    last_chunk_words = chunks[-1].split()[-overlap:]
                    current_chunk.extend(last_chunk_words)
                    current_size = len(last_chunk_words)

            current_chunk.extend(words)
            current_size += sentence_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_pdf_to_json(self, pdf_path, output_json_path, chunk_size=300, overlap=30):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"الملف {pdf_path} غير موجود")

        print("جاري استخراج النص من PDF...")
        text = self.extract_text_from_pdf(pdf_path)

        if not text.strip():
            raise ValueError("لم يتم العثور على نص في ملف PDF")

        print("جاري تنظيف النص...")
        cleaned_text = self.clean_arabic_text(text)

        if not cleaned_text.strip():
            raise ValueError("النص فارغ بعد التنظيف")

        print("جاري تقسيم النص إلى chunks...")
        chunks = self.split_into_chunks(cleaned_text, chunk_size, overlap)

        if not chunks:
            words = cleaned_text.split()
            chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        if not chunks:
            raise ValueError("لم يتم إنشاء أي chunks من النص")

        output_data = {
            "metadata": {
                "source": pdf_path,
                "total_chunks": len(chunks),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "language": "arabic",
                "total_words": sum(len(chunk.split()) for chunk in chunks)
            },
            "chunks": [
                {
                    "id": i + 1,
                    "text": chunk,
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk)
                } for i, chunk in enumerate(chunks)
            ]
        }

        os.makedirs(os.path.dirname(output_json_path) if os.path.dirname(output_json_path) else '.', exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"تم الانتهاء! تم حفظ النتائج في: {output_json_path}")
        print(f"عدد الكتل النصية: {len(chunks)}")
        print(f"إجمالي الكلمات: {output_data['metadata']['total_words']}")

        return output_data


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("data.ui", self)

        # زر اختيار الملف
        self.pushButton_browse.clicked.connect(self.browse_file)

        # زر تشغيل المعالجة
        self.pushButton_2.clicked.connect(self.run_processing)

        # كائن المعالجة
        self.processor = ArabicPDFProcessor()

        self._stdout_backup = sys.stdout
        sys.stdout = self

    def write(self, msg):
        self.plainTextEdit.appendPlainText(msg)
        self._stdout_backup.write(msg)

    def flush(self):
        pass
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "اختر ملف PDF",
            "",
            "PDF Files (*.pdf);;كل الملفات (*.*)"
        )

        if file_path:
            self.lineEdit_path.setText(file_path)
            print("تم اختيار الملف:", file_path)

    def run_processing(self):
        pdf_path = self.lineEdit_path.text().strip()
        if not pdf_path or not os.path.exists(pdf_path):
            QMessageBox.warning(self, "خطأ", "يرجى اختيار ملف PDF صالح!")
            return

        # اختيار مكان حفظ JSON
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "اختر مكان حفظ output.json",
            "output.json",
            "JSON Files (*.json);;كل الملفات (*.*)"
        )
        if not output_path:
            print("تم إلغاء اختيار مكان الحفظ.")
            return

        try:
            print("جاري معالجة PDF...")
            self.processor.process_pdf_to_json(pdf_path, output_path)
            QMessageBox.information(self, "تم", f"تمت المعالجة وحفظ النتائج في:\n{output_path}")
        except Exception as e:
            print(f"حدث خطأ أثناء المعالجة: {e}")
            QMessageBox.critical(self, "خطأ", str(e))


# =========================
# SECOND WINDOW (data.ui)
# =========================
class SecondWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("data.ui", self)

        self.processor = ArabicPDFProcessor()

        self.pushButton_browse.clicked.connect(self.browse_file)
        self.pushButton_2.clicked.connect(self.run_processing)

        # redirect print to UI
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, msg):
        self.plainTextEdit.appendPlainText(msg)
        self._stdout.write(msg)

    def flush(self):
        pass

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "اختر ملف PDF", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.lineEdit_path.setText(file_path)
            print("تم اختيار الملف:", file_path)

    def run_processing(self):
        pdf_path = self.lineEdit_path.text().strip()
        if not os.path.exists(pdf_path):
            QMessageBox.warning(self, "خطأ", "ملف PDF غير صالح")
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "حفظ JSON", "output.json", "JSON Files (*.json)"
        )
        if not output_path:
            return

        try:
            self.processor.process_pdf_to_json(pdf_path, output_path)
            QMessageBox.information(self, "تم", "تمت المعالجة بنجاح")
        except Exception as e:
            QMessageBox.critical(self, "خطأ", str(e))



def run_streamlit():
    app_path = os.path.abspath("app.py")   # اسم ملف Streamlit
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path],
        shell=True
    )

# =========================
# MAIN WINDOW (main.ui)
# =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)

        self.secondaryButton_2.clicked.connect(self.open_second)
        self.pushButton.clicked.connect(run_streamlit)


    def open_second(self):
        self.second = SecondWindow()
        self.second.show()
        self.close()  # اختياري


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
