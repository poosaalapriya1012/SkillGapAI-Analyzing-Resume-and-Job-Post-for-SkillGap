"""
Task 2: POS Tagging (20 points)
Includes:
2.1 Basic POS Tagging
2.2 Extract Nouns Only
2.3 Identify Skill Patterns
"""

import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------------------
# Question 2.1: Basic POS Tagging (7 points)
# ------------------------------------------------------------
def pos_tag_resume(text):
    """
    Tags each word in the text with its Part of Speech (POS)
    Returns a list of tuples: (word, POS_tag)
    Uses spaCy for tagging
    """
    doc = nlp(text)
    tagged_words = [(token.text, token.pos_) for token in doc]
    return tagged_words


# ------------------------------------------------------------
# Question 2.2: Extract Nouns Only (7 points)
# ------------------------------------------------------------
def extract_nouns(text):
    """
    Extracts all NOUN and PROPN (proper nouns) from the text.
    Returns a list of nouns — often representing skills.
    """
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return nouns


# ------------------------------------------------------------
# Question 2.3: Identify Skill Patterns (6 points)
# ------------------------------------------------------------
def find_adj_noun_patterns(text):
    """
    Finds all 'Adjective + Noun' combinations (skill patterns).
    Example: 'Machine Learning', 'Deep Learning'
    Returns list of such patterns.
    """
    doc = nlp(text)
    patterns = []
    
    for i in range(len(doc) - 1):
        if doc[i].pos_ == "ADJ" and doc[i + 1].pos_ == "NOUN":
            patterns.append(f"{doc[i].text} {doc[i + 1].text}")
        elif doc[i].pos_ == "PROPN" and doc[i + 1].pos_ == "PROPN":
            # For cases like "Machine Learning" or "Natural Language"
            patterns.append(f"{doc[i].text} {doc[i + 1].text}")
    
    return patterns


# ------------------------------------------------------------
# Testing All Functions
# ------------------------------------------------------------
if __name__ == "__main__":
    # Test for Q2.1
    text1 = "John is an experienced Python developer"
    print("Q2.1 Output:")
    print(pos_tag_resume(text1))
    print()

    # Test for Q2.2
    text2 = "Experienced Data Scientist proficient in Machine Learning and Python programming"
    print("Q2.2 Output:")
    print(extract_nouns(text2))
    print()

    # Test for Q2.3
    text3 = "Expert in Machine Learning, Deep Learning, and Natural Language Processing"
    print("Q2.3 Output:")
    print(find_adj_noun_patterns(text3))
    print()
