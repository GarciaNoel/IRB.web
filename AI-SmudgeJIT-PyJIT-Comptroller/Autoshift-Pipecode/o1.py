import sys
import ollama
import re
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QInputDialog, QTabWidget
)
from PyQt5.QtGui import QTextDocument, QColor, QTextCursor, QTextCharFormat
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QComboBox

MODEL = "deepseek-r1:1.5b"

class TabPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        # Instruction label
        self.instruction = QLabel("Send prompt")
        # Dropdown for Ask button function
        self.ask_mode = QComboBox()
        self.ask_mode.addItems(["With History", "No Memory", "There is some"])

        # Chat area
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask Agent...")
        self.button = QPushButton("Ask")
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        # Editors layout
        self.editors_layout = QHBoxLayout()
        self.editor_label = QLabel("Text Editor:")
        self.editor = QTextEdit()
        self.highlights_label = QLabel("Current Highlight(s):")
        self.highlights_editor = QTextEdit()
        self.highlights_list = QListWidget()
        self.highlights = []  # List of (start, end, text)
        self.current_highlight_index = None

        self.ask_mode.currentIndexChanged.connect(self.update_ask_mode)
        self.ask_function = self.ask_ollama

        # Layout setup
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.editor_label)
        left_layout.addWidget(self.editor)
        self.editors_layout.addLayout(left_layout)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.highlights_label)
        right_layout.addWidget(self.highlights_editor)
        right_layout.addWidget(self.highlights_list)
        self.editors_layout.addLayout(right_layout)

        # Buttons
        self.save_button = QPushButton("Save Text File")
        self.obfuscate_button = QPushButton("Obfuscate & Remove Comments")
        self.plaintext_button = QPushButton("Convert to Plain Text")
        self.new_highlight_button = QPushButton("New Highlight")

        # Horizontal layout for buttons
        self.button_row = QHBoxLayout()
        self.button_row.addWidget(self.save_button)
        self.button_row.addWidget(self.obfuscate_button)
        self.button_row.addWidget(self.plaintext_button)
        self.button_row.addWidget(self.new_highlight_button)

        # Assemble
        self.layout.addWidget(self.ask_mode)
        self.layout.addWidget(self.instruction)


        self.layout.addWidget(self.input)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.output)
        self.layout.addLayout(self.editors_layout)
        self.layout.addLayout(self.button_row)  # Use horizontal layout for buttons
        self.setLayout(self.layout)

        # Connect
        self.button.clicked.connect(self.on_ask)
        self.save_button.clicked.connect(self.save_text_file)
        self.obfuscate_button.clicked.connect(self.obfuscate_and_remove_comments)
        self.plaintext_button.clicked.connect(self.convert_to_plain_text)
        self.new_highlight_button.clicked.connect(self.highlight_selection)
        self.highlights_list.itemDoubleClicked.connect(self.select_highlight_for_edit)
        self.highlights_editor.textChanged.connect(self.update_highlight_from_editor)
        self.history = []

    def highlight_selection(self):
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            start = cursor.selectionStart()
            end = cursor.selectionEnd()
            selected_text = cursor.selectedText()
            # Highlight in editor
            fmt = QTextCharFormat()
            fmt.setBackground(QColor("yellow"))
            cursor.mergeCharFormat(fmt)
            # Track highlight
            self.highlights.append((start, end, selected_text))
            item = QListWidgetItem(selected_text)
            self.highlights_list.addItem(item)
            # Set right editor to this highlight and remember index
            self.current_highlight_index = len(self.highlights) - 1
            self.highlights_editor.blockSignals(True)
            self.highlights_editor.setPlainText(selected_text)
            self.highlights_editor.blockSignals(False)

    def select_highlight_for_edit(self, item):
        idx = self.highlights_list.row(item)
        start, end, text = self.highlights[idx]
        self.current_highlight_index = idx
        self.highlights_editor.blockSignals(True)
        self.highlights_editor.setPlainText(text)
        self.highlights_editor.blockSignals(False)

    def update_highlight_from_editor(self):
        idx = self.current_highlight_index
        if idx is None or idx >= len(self.highlights):
            return
        start, end, old_text = self.highlights[idx]
        new_text = self.highlights_editor.toPlainText()
        # Update left editor
        cursor = self.editor.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(new_text)
        # Re-highlight the new text
        cursor.setPosition(start)
        cursor.setPosition(start + len(new_text), QTextCursor.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("yellow"))
        cursor.mergeCharFormat(fmt)
        # Update highlight tracking and list
        self.highlights[idx] = (start, start + len(new_text), new_text)
        self.highlights_list.item(idx).setText(new_text)

    def convert_to_plain_text(self):
        html = self.editor.toHtml()
        doc = QTextDocument()
        doc.setHtml(html)
        self.editor.clear()
        self.editor.setTextColor(Qt.black)
        self.editor.setPlainText(doc.toPlainText())

    def ask_ollama(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        response = ollama.chat(model=MODEL, messages=self.history)
        answer = response['message']['content']
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def ask_ollama_there(self, prompt):
        prompt = "There is some answer x to \"" + prompt + "\" and it is,"
        self.history.append({"role": "user", "content": prompt})
        response = ollama.chat(model=MODEL, messages=self.history)
        answer = response['message']['content']
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def update_ask_mode(self, idx):
        if self.ask_mode.currentText() == "Chat":
            self.ask_function = self.ask_ollama
        elif self.ask_mode.currentText() == "There is some":
            self.ask_function = self.ask_ollama_there
        else:
            self.ask_function = self.ask_ollama_completion

    def ask_ollama_completion(self, prompt):
        # Example: use ollama.generate instead of ollama.chat
        response = ollama.generate(model=MODEL, prompt=prompt)
        return response['response']

    def on_ask(self):
        prompt = self.input.text()
        if prompt.strip():
            self.output.append(f"You: {prompt}")
            answer = self.ask_function(prompt)
            self.output.append(f"Ollama: {answer}\n")
            self.input.clear()

    def save_text_file(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Text File", "", "Text Files (*.txt);;All Files (*)", options=options)
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.editor.toPlainText())

    def obfuscate_and_remove_comments(self):
        text = self.editor.toPlainText()
        lines = text.splitlines()
        non_comment_lines = [line for line in lines if not line.strip().startswith("#")]
        code = "\n".join(non_comment_lines)
        name_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')
        names = {}
        name_counter = 1

        def replace_name(match):
            nonlocal name_counter
            name = match.group(1)
            if name in {"def", "class", "import", "from", "as", "if", "else", "elif", "for", "while", "return", "with", "try", "except", "finally", "in", "is", "not", "and", "or", "pass", "break", "continue", "lambda", "yield", "global", "nonlocal", "assert", "del", "raise", "True", "False", "None"}:
                return name
            if name not in names:
                names[name] = f"NAME{name_counter}"
                name_counter += 1
            return names[name]

        obfuscated_code = name_pattern.sub(replace_name, code)
        self.editor.setPlainText(obfuscated_code)

    def get_tab_data(self):
        return {
            "text": self.editor.toPlainText(),
            "highlights": self.highlights,
            "history": self.history
        }

    def set_tab_data(self, data):
        self.editor.setPlainText(data.get("text", ""))
        self.highlights = data.get("highlights", [])
        self.history = data.get("history", [])
        self.highlights_list.clear()
        for _, _, txt in self.highlights:
            self.highlights_list.addItem(QListWidgetItem(txt))
        self.highlights_editor.clear()
        self.current_highlight_index = None

class OllamaWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama PyQt Frontend")
        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        

        # Session buttons
        session_layout = QHBoxLayout()
        self.save_session_button = QPushButton("Save Session")
        self.load_session_button = QPushButton("Load Session")
        self.add_tab_button = QPushButton("Add Tab")
        session_layout.addWidget(self.save_session_button)
        session_layout.addWidget(self.load_session_button)
        session_layout.addWidget(self.add_tab_button)
        self.layout.addLayout(session_layout)
        self.setLayout(self.layout)

        self.layout.addWidget(self.tabs)

        self.save_session_button.clicked.connect(self.save_session)
        self.load_session_button.clicked.connect(self.load_session)
        self.add_tab_button.clicked.connect(self.add_tab)

        # Start with one tab
        self.add_tab()

    def add_tab(self):
        page = TabPage()
        idx = self.tabs.addTab(page, f"Page {self.tabs.count() + 1}")
        self.tabs.setCurrentIndex(idx)

    def save_session(self):
        session = []
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            session.append(tab.get_tab_data())
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "JSON Files (*.json);;All Files (*)", options=options)
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(session, f)

    def load_session(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Session", "", "JSON Files (*.json);;All Files (*)", options=options)
        if filename:
            with open(filename, "r", encoding="utf-8") as f:
                session = json.load(f)
            self.tabs.clear()
            for tab_data in session:
                page = TabPage()
                page.set_tab_data(tab_data)
                self.tabs.addTab(page, f"Page {self.tabs.count() + 1}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OllamaWindow()
    window.show()
    sys.exit(app.exec_())