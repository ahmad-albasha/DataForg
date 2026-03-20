import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # تحميل واجهة Qt Designer من الملف data.ui
        uic.loadUi("data.ui", self)

        # ربط الزر بالدالة
        # تأكد أن اسم الزر في الـ Designer هو browseButton
        self.browseButton.clicked.connect(self.browse_file)

    def browse_file(self):
        # فتح نافذة اختيار ملف
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "اختر ملف",               # عنوان النافذة
            "",                        # مجلد البداية
            "كل الملفات (*.*);;ملفات نصية (*.txt);;PDF Files (*.pdf)"
        )

        if file_path:
            # عرض مسار الملف في الـ QLineEdit
            # تأكد أن اسم حقل النص في الـ Designer هو filePathLineEdit
            self.filePathLineEdit.setText(file_path)

            # فيك تشتغل على الملف هون
            print("تم اختيار الملف:", file_path)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
