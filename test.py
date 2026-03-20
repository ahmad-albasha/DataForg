from PyQt5.QtWidgets import QApplication, QMessageBox
import sys

app = QApplication(sys.argv)
QMessageBox.information(None, "Test", "PyQt5 Works!")
sys.exit(app.exec_())
