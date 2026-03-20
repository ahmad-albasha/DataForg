import sys
import os
import json
import re
import subprocess
import io

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import fitz
import pytesseract
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image


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
            r'C:\poppler\Library\bin',
            r'C:\Program Files\poppler\bin',
            r'C:\Program Files (x86)\poppler\bin',
            '/usr/bin',
            '/usr/local/bin',
            '/opt/homebrew/bin',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
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
                text = self._extract_text_from_scanned_pdf(pdf_path)
        except Exception:
            text = self._extract_text_from_scanned_pdf(pdf_path)
        return text

    def _extract_text_from_scanned_pdf(self, pdf_path):
        text = self._extract_text_with_pymupdf(pdf_path)
        if not text.strip():
            text = self._extract_text_with_pdf2image(pdf_path)
        return text

    def _extract_text_with_pymupdf(self, pdf_path):
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                    continue
                images = page.get_images()
                for img in images:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    try:
                        img_text = pytesseract.image_to_string(image, lang='ara')
                        text += img_text + "\n"
                    except:
                        img_text = pytesseract.image_to_string(image, lang='eng')
                        text += img_text + "\n"
            doc.close()
        except Exception:
            pass
        return text

    def _extract_text_with_pdf2image(self, pdf_path):
        text = ""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=200, poppler_path=self.poppler_path)
            for image in images:
                try:
                    page_text = pytesseract.image_to_string(image, lang='ara')
                    text += page_text + "\n"
                except:
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    text += page_text + "\n"
        except Exception:
            pass
        return text

    def clean_arabic_text(self, text):
        if not text.strip():
            return ""
        text = re.sub(r'[^\u0600-\u06FF\s\.\,\!\?\:\;\(\)\[\]\{\}0-9a-zA-Z]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        try:
            text = arabic_reshaper.reshape(text, self.reshaping_config)
            text = get_display(text)
        except:
            pass
        return text

    def split_into_chunks(self, text, chunk_size=300, overlap=30):
        if not text.strip():
            return []
        sentences = re.split(r'[.!?؟۔]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_chunk = []
        current_size = 0
        for sentence in sentences:
            words = sentence.split()
            if current_size + len(words) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_size = len(current_chunk)
            current_chunk.extend(words)
            current_size += len(words)
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def process_pdf_to_json(self, pdf_path, output_json_path, chunk_size=300, overlap=30):
        text = self.extract_text_from_pdf(pdf_path)
        cleaned_text = self.clean_arabic_text(text)
        chunks = self.split_into_chunks(cleaned_text, chunk_size, overlap)
        if not chunks:
            words = cleaned_text.split()
            chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
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
                {"id": i + 1, "text": chunk, "word_count": len(chunk.split()), "char_count": len(chunk)}
                for i, chunk in enumerate(chunks)
            ]
        }
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        return output_data


class WorkerThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, processor, pdf_path, output_path):
        super().__init__()
        self.processor = processor
        self.pdf_path = pdf_path
        self.output_path = output_path

    def run(self):
        try:
            self.progress.emit("جاري معالجة PDF...")
            self.processor.process_pdf_to_json(self.pdf_path, self.output_path)
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("data.ui", self)

        self.pushButton_browse.clicked.connect(self.browse_file)
        self.pushButton_2.clicked.connect(self.run_processing)

        self.processor = ArabicPDFProcessor()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "اختر ملف PDF", "", "PDF Files (*.pdf);;كل الملفات (*.*)")
        if file_path:
            self.lineEdit_path.setText(file_path)
            self.plainTextEdit.appendPlainText(f"تم اختيار الملف: {file_path}")

    def run_processing(self):
        pdf_path = self.lineEdit_path.text().strip()
        if not pdf_path or not os.path.exists(pdf_path):
            QMessageBox.warning(self, "خطأ", "يرجى اختيار ملف PDF صالح!")
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "اختر مكان حفظ output.json", "output.json", "JSON Files (*.json);;كل الملفات (*.*)")
        if not output_path:
            self.plainTextEdit.appendPlainText("تم إلغاء اختيار مكان الحفظ.")
            return

        self.thread = WorkerThread(self.processor, pdf_path, output_path)
        self.thread.progress.connect(lambda msg: self.plainTextEdit.appendPlainText(msg))
        self.thread.finished.connect(lambda path: QMessageBox.information(self, "تم", f"تمت المعالجة وحفظ النتائج في:\n{path}"))
        self.thread.error.connect(lambda e: QMessageBox.critical(self, "خطأ", e))
        self.thread.start()
