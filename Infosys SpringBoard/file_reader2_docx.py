import os
from pathlib import Path
from docx import Document   

cwd = os.getcwd()

file_path = Path(cwd) / "priyapoosaala_resume.docx"


if file_path.exists():
    doc = Document(file_path)

    print("Word Document Content:\n")
    for para in doc.paragraphs:
        if para.text.strip():
            print(para.text)
else:
    print(f"File not found at: {file_path}")