import os
from pathlib import Path

cwd = os.getcwd()
file_path = Path(cwd) / "priyapoosaala_resume.txt"

if file_path.exists():
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    print("Full content:\n")
    print(content)
else:
    print(f"File not found at: {file_path}")
