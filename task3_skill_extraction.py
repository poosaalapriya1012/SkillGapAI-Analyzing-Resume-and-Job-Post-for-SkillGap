"""
Task 3: Skill Extraction (25 points)
Includes:
3.1 Create Skill Database
3.2 Simple Skill Matcher
3.3 Handle Skill Abbreviations
"""

import re

# ------------------------------------------------------------
# Question 3.1: Create Skill Database (8 points)
# ------------------------------------------------------------

SKILL_DATABASE = {
    'programming_languages': [
        'Python', 'Java', 'C', 'C++', 'C#', 'JavaScript', 'R', 'Go', 'Swift', 'PHP'
    ],
    'frameworks': [
        'TensorFlow', 'PyTorch', 'React', 'Angular', 'Django', 'Flask', 'Spring', 'Node.js', 'Keras', 'Bootstrap'
    ],
    'databases': [
        'MySQL', 'PostgreSQL', 'MongoDB', 'SQLite', 'Oracle'
    ],
    'cloud': [
        'AWS', 'Azure', 'Google Cloud Platform', 'GCP', 'IBM Cloud'
    ],
    'soft_skills': [
        'Leadership', 'Communication', 'Teamwork', 'Problem Solving', 'Adaptability'
    ]
}

# ------------------------------------------------------------
# Question 3.2: Simple Skill Matcher (8 points)
# ------------------------------------------------------------
def extract_skills(text, skill_database):
    """
    Searches for skills from SKILL_DATABASE in the given text.
    Case-insensitive matching.
    Returns dictionary categorized by skill type.
    """
    found_skills = {}

    for category, skills in skill_database.items():
        matches = []
        for skill in skills:
            # Use regex for case-insensitive word boundary matching
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, flags=re.IGNORECASE):
                matches.append(skill)
        if matches:
            found_skills[category] = matches

    return found_skills


# ------------------------------------------------------------
# Question 3.3: Handle Skill Abbreviations (9 points)
# ------------------------------------------------------------

# Abbreviation dictionary
ABBREVIATIONS = {
    'ML': 'Machine Learning',
    'DL': 'Deep Learning',
    'NLP': 'Natural Language Processing',
    'AI': 'Artificial Intelligence',
    'DS': 'Data Science',
    'JS': 'JavaScript',
    'K8s': 'Kubernetes',
    'GCP': 'Google Cloud Platform',
    'DB': 'Database',
    'CV': 'Computer Vision',
    'UI': 'User Interface',
    'UX': 'User Experience'
}

def normalize_skills(skill_list):
    """
    Converts abbreviations in a list to their full names
    based on ABBREVIATIONS dictionary.
    If abbreviation not found, keeps original skill name.
    """
    normalized = [ABBREVIATIONS.get(skill, skill) for skill in skill_list]
    return normalized


# ------------------------------------------------------------
# Testing All Functions
# ------------------------------------------------------------
if __name__ == "__main__":
    # Test for Q3.1 (Database)
    print("Q3.1 Output (Skill Categories):")
    for key, value in SKILL_DATABASE.items():
        print(f"{key}: {value}")
    print()

    # Test for Q3.2 (Skill Extraction)
    text = "Proficient in Python, Java, TensorFlow, and AWS. Strong leadership skills."
    print("Q3.2 Output (Extracted Skills):")
    print(extract_skills(text, SKILL_DATABASE))
    print()

    # Test for Q3.3 (Abbreviation Normalization)
    skills = ['ML', 'DL', 'NLP', 'JS', 'K8s', 'AWS', 'GCP']
    print("Q3.3 Output (Normalized Skills):")
    print(normalize_skills(skills))
    print()
