"""
Task 1: Text Preprocessing (20 points)
Includes:
1.1 Basic Text Cleaning
1.2 Tokenization
1.3 Stop Words Removal
1.4 Lemmatization
"""

import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------------------
# Question 1.1: Basic Text Cleaning (5 points)
# ------------------------------------------------------------
def clean_resume_text(text):
    """
    Removes:
    - All email addresses
    - All phone numbers
    - All URLs
    - Special characters except (+ # - .)
    Converts text to lowercase
    """
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers (like +1-555-0123 or 555-0123)
    text = re.sub(r'\+?\d[\d\-\s()]{7,}\d', '', text)
    
    # Remove URLs (like www.example.com or http://example.com)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove unwanted special characters except (+ # - .)
    text = re.sub(r'[^a-zA-Z0-9\s+#\-\.+]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ------------------------------------------------------------
# Question 1.2: Tokenization (5 points)
# ------------------------------------------------------------
def tokenize_text(text):
    """
    Splits text into individual words (tokens) using spaCy
    Returns a list of tokens
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens


# ------------------------------------------------------------
# Question 1.3: Stop Words Removal (5 points)
# ------------------------------------------------------------
def remove_stop_words(text):
    """
    Removes common stop words while preserving programming
    language names: C, R, Go, D
    Returns cleaned text (string)
    """
    preserve = {'c', 'r', 'go', 'd'}
    doc = nlp(text)
    filtered_words = [token.text for token in doc
                      if token.text.lower() not in STOP_WORDS or token.text.lower() in preserve]
    return ' '.join(filtered_words)


# ------------------------------------------------------------
# Question 1.4: Lemmatization (5 points)
# ------------------------------------------------------------
def lemmatize_text(text):
    """
    Converts words to their base (lemma) form using spaCy
    Returns lemmatized text (string)
    """
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc]
    return ' '.join(lemmatized)


# ------------------------------------------------------------
# Testing All Functions
# ------------------------------------------------------------
if __name__ == "__main__":
    # Test for Q1.1
    text1 = """
    Contact: john@email.com | Phone: +1-555-0123
    Visit: www.johndoe.com
    Skills: Python, C++, C#, .NET
    """
    print("Q1.1 Output:")
    print(clean_resume_text(text1))
    print()

    # Test for Q1.2
    text2 = "I'm a Python developer. I've worked on ML projects."
    print("Q1.2 Output:")
    print(tokenize_text(text2))
    print()

    # Test for Q1.3
    text3 = "I have experience in Python and R programming with excellent skills in C and Go"
    print("Q1.3 Output:")
    print(remove_stop_words(text3))
    print()

    # Test for Q1.4
    text4 = "I am working on developing multiple applications using programming languages"
    print("Q1.4 Output:")
    print(lemmatize_text(text4))
    print()
