
import os
from pathlib import Path
import PyPDF2

cwd = os.getcwd()

file_path = Path(cwd) / "priyapoosaala_resume.pdf"


if file_path.exists():
    with open(file_path, "rb") as pdf_file:  
        reader = PyPDF2.PdfReader(pdf_file)
        
        print("PDF Content:\n")
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            print(f"\n--- Page {page_num} ---\n")
            print(text)
else:
    print(f"File not found at: {file_path}")