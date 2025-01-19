import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QCursor

widgets = {
    "logo": [],
    "button": [],
    "score": [],
    "question": [],
    "answer1": [],
    "answer2": [],
    "answer3": [],
    "answer4": [],
}    

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Who wants to be a programmer??")
window.setFixedWidth(1000)

window.setStyleSheet("background: #161219;") # CSS

grid = QGridLayout()

def clear_widgets():
    for widget in widgets:
        if widgets[widget] != []:
            widgets[widget][-1].hide()
        for i in range(len(widgets[widget])):
            widgets[widget].pop()

def show_frame1():
    clear_widgets()
    frame1()    

def start_game():
    clear_widgets()
    frame2()




def create_buttons(answer, l_margin, r_margin):
    button = QPushButton(answer)
    button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
    button.setFixedWidth(485)
    button.setStyleSheet(f"""
        *{{border: 4px solid '#BC006C';
        margin-left: {l_margin}px;
        margin-right: {r_margin}px;
        color: 'white';
        font-size: 16px;
        border-radius: 25px;
        padding: 15px 0;
        margin-top: 20px;
        }}
        *:hover{{background: '#BC006C';}}

    """)
    button.clicked.connect(show_frame1)

    return button


def frame1():
    # display logo
    image = QPixmap("logo.png")
    logo = QLabel()
    logo.setPixmap(image)
    logo.setAlignment(QtCore.Qt.AlignCenter)
    logo.setStyleSheet("margin-top: 100px;")
    widgets["logo"].append(logo)

    #button widget
    button = QPushButton("PLAY")
    button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
    button.setStyleSheet("""
        *{border: 4px solid '#BC006C';
        border-radius:45px;
        font-size: 35px;
        color: 'white'; 
        padding: 25px 0;
        margin: 100px 200px;}

        *:hover{background: '#BC006C';}
        
    """)
    button.clicked.connect(start_game)


    widgets["button"].append(button)

    grid.addWidget(widgets['logo'][-1], 0, 0, 1, 2)
    grid.addWidget(widgets['button'][-1], 1, 0, 1, 2)

# frame1()

def frame2():
    score = QLabel("80")
    score.setAlignment(QtCore.Qt.AlignCenter)
    score.setStyleSheet(
        '''
        font-size: 35px;
        color: 'white';
        padding: 15px 10px;
        margin: 20px 200px;
        background: '#64A314';
        border: 1px solid '#64A314';
        border-radius: 35px;
        text-align: center;
        '''
    )
    widgets["score"].append(score)


    question = QLabel("Placeholder text will go here")
    question.setAlignment(QtCore.Qt.AlignCenter)
    question.setWordWrap(True)
    question.setStyleSheet("""
        font-size: 25px;
        color: 'white';
        padding: 75px;
    """)
    widgets["question"].append(question)

    button1 = create_buttons("Answer 1",85,5)
    button2 = create_buttons("Answer 2",5,85)
    button3 = create_buttons("Answer 3",85,5)
    button4 = create_buttons("Answer 4",5,85)

    widgets["answer1"].append(button1)
    widgets["answer2"].append(button2)
    widgets["answer3"].append(button3)
    widgets["answer4"].append(button4)
 
    image = QPixmap("logo_bottom.png")
    logo = QLabel()
    logo.setPixmap(image)
    logo.setAlignment(QtCore.Qt.AlignCenter)
    logo.setStyleSheet("margin-top: 75px; margin-bottom: 30px;")
    widgets["logo"].append(logo)
    

    grid.addWidget(widgets['score'][-1], 0, 1)
    grid.addWidget(widgets['question'][-1], 1, 0, 1, 2) #row, col, rowSpan, colSpan
    grid.addWidget(widgets['answer1'][-1], 2, 0)
    grid.addWidget(widgets['answer2'][-1], 2, 1)
    grid.addWidget(widgets['answer3'][-1], 3, 0)
    grid.addWidget(widgets['answer4'][-1], 3, 1)
    grid.addWidget(widgets['logo'][-1], 4, 0, 1, 2)



frame1()

window.setLayout(grid)

window.show()

sys.exit(app.exec())



