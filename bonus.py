# ==========================
# Bonus Questions (10 points)
# ==========================

import re
import spacy
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------
# Bonus 1: Skill Frequency Counter (5 points)
# ---------------------------------------
def count_skill_frequency(text):
    """
    Counts how many times each skill appears in the text.
    Matches both single-word and multi-word skills.
    Returns a dictionary sorted by frequency.
    """

    # Define a sample skill list (you can extend this from your SKILL_DATABASE)
    skills = [
        "Python", "Machine Learning", "Deep Learning", "Java", "JavaScript", 
        "TensorFlow", "React", "Django", "Data Science"
    ]
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count occurrences
    counts = {}
    for skill in skills:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            counts[skill] = len(matches)
    
    # Sort by frequency (descending)
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    # Display results
    for skill, freq in sorted_counts.items():
        print(f"{skill}: {freq} times")
    
    return sorted_counts


# ---------------------------------------
# Bonus 2: Skill Context Extractor (5 points)
# ---------------------------------------
def extract_skill_context(text, skill):
    """
    Finds all sentences containing the given skill.
    Returns context sentences around that skill.
    """

    doc = nlp(text)
    skill_lower = skill.lower()
    contexts = []

    for sent in doc.sents:
        if skill_lower in sent.text.lower():
            contexts.append(sent.text.strip())

    # Display results
    print(f"Skill: {skill}")
    for i, ctx in enumerate(contexts, 1):
        print(f"Context {i}: \"{ctx}\"")

    return contexts


# ---------------------------------------
# Test Both Bonus Functions
# ---------------------------------------
if __name__ == "__main__":
    print("=== BONUS 1: Skill Frequency Counter ===\n")
    text1 = """
    Python developer with Python experience. 
    Used Python and Machine Learning. 
    Machine Learning projects with Python.
    """
    count_skill_frequency(text1)

    print("\n=== BONUS 2: Skill Context Extractor ===\n")
    text2 = """
    I am a Python developer. I have 5 years of experience in Python.
    Also worked on Java projects. Python is my primary language.
    """
    extract_skill_context(text2, "Python")
