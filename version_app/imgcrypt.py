import sys
from PyQt5.QtWidgets import QApplication, QWidget

class App(QWidget):
    def __init__(self, size):
        super.__init__()
        self.title = "cute ass anniversary"
        self.screen_x, self.screen_y = size
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(0,0,400,400)

        ##show/store option

        self.show()

    # scanning camera ui

    #

def start_app():
    app = QApplication(sys.argv)
    size = app.primaryScreen().size()
    main = App(size)
    sys.exit(app.exec_())