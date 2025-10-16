"""
Task 4: Named Entity Recognition (15 points)
Includes:
4.1 Extract Named Entities
4.2 Annotate Training Data
"""

import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------------------
# Question 4.1: Extract Named Entities (8 points)
# ------------------------------------------------------------
def extract_entities(text):
    """
    Uses spaCy's Named Entity Recognition (NER)
    Extracts entities of types: ORG, PERSON, GPE, PRODUCT
    Returns list of tuples (entity_text, label)
    """
    doc = nlp(text)
    selected_labels = {"ORG", "PERSON", "GPE", "PRODUCT"}
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in selected_labels]
    return entities


# ------------------------------------------------------------
# Question 4.2: Annotate Training Data (7 points)
# ------------------------------------------------------------
"""
Given Sentences:
sentences = [
    "Python developer with 5 years of experience",
    "Expert in Machine Learning and Data Science",
    "Proficient in TensorFlow and PyTorch frameworks",
    "Strong SQL and MongoDB database skills",
    "Excellent communication and leadership abilities"
]

We manually find start & end positions for each skill word/phrase.
Character indexing starts at 0.
"""

# 1. "Python developer with 5 years of experience"
# "Python" = start: 0, end: 6

# 2. "Expert in Machine Learning and Data Science"
# "Machine Learning" = start: 10, end: 26
# "Data Science"     = start: 31, end: 43

# 3. "Proficient in TensorFlow and PyTorch frameworks"
# "TensorFlow" = start: 14, end: 24
# "PyTorch"    = start: 29, end: 36

# 4. "Strong SQL and MongoDB database skills"
# "SQL"     = start: 7, end: 10
# "MongoDB" = start: 15, end: 22

# 5. "Excellent communication and leadership abilities"
# "communication" = start: 10, end: 23
# "leadership"    = start: 28, end: 38

TRAIN_DATA = [
    ("Python developer with 5 years of experience",
     {"entities": [(0, 6, "SKILL")]}),
    
    ("Expert in Machine Learning and Data Science",
     {"entities": [(10, 26, "SKILL"), (31, 43, "SKILL")]}),
    
    ("Proficient in TensorFlow and PyTorch frameworks",
     {"entities": [(14, 24, "SKILL"), (29, 36, "SKILL")]}),
    
    ("Strong SQL and MongoDB database skills",
     {"entities": [(7, 10, "SKILL"), (15, 22, "SKILL")]}),
    
    ("Excellent communication and leadership abilities",
     {"entities": [(10, 23, "SKILL"), (28, 38, "SKILL")]})
]


# ------------------------------------------------------------
# Testing All Functions
# ------------------------------------------------------------
if __name__ == "__main__":
    # Test for Q4.1
    text = "John worked at Google and Microsoft in New York. He used TensorFlow and Python."
    print("Q4.1 Output (Extracted Entities):")
    print(extract_entities(text))
    print()

    # Test for Q4.2
    print("Q4.2 Output (Annotated Training Data):")
    for data in TRAIN_DATA:
        print(data)
