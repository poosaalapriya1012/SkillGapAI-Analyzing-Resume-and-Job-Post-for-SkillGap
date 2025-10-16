# ==========================
# Task 5: Complete Skill Extractor
# ==========================

import re
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------
# Skill Database (reused from previous task)
# ---------------------------------------
SKILL_DATABASE = {
    'programming_languages': ['Python', 'Java', 'C++', 'C#', 'JavaScript', 'Ruby', 'Go', 'Swift', 'PHP', 'R'],
    'frameworks': ['TensorFlow', 'PyTorch', 'React', 'Angular', 'Django', 'Flask', 'Spring', 'Keras', 'Vue', '.NET'],
    'databases': ['MySQL', 'PostgreSQL', 'MongoDB', 'SQLite', 'Oracle'],
    'cloud': ['AWS', 'Azure', 'Google Cloud Platform', 'GCP', 'IBM Cloud'],
    'soft_skills': ['Leadership', 'Teamwork', 'Communication', 'Problem-solving', 'Analytical']
}

# ---------------------------------------
# Helper Function 1: Keyword Matching
# ---------------------------------------
def extract_skills(text, skill_database):
    found_skills = {category: [] for category in skill_database.keys()}
    text_lower = text.lower()

    for category, skills in skill_database.items():
        for skill in skills:
            if skill.lower() in text_lower:
                found_skills[category].append(skill)
    
    # Remove empty categories
    found_skills = {k: v for k, v in found_skills.items() if v}
    return found_skills

# ---------------------------------------
# Helper Function 2: POS Pattern Matching
# Finds ADJ + NOUN and NOUN + NOUN patterns
# ---------------------------------------
def find_skill_patterns(text):
    doc = nlp(text)
    patterns = set()

    for i in range(len(doc) - 1):
        word1, word2 = doc[i], doc[i+1]

        # Pattern 1: Adjective + Noun (e.g., Machine Learning)
        if word1.pos_ == "ADJ" and word2.pos_ == "NOUN":
            patterns.add(f"{word1.text} {word2.text}")

        # Pattern 2: Noun + Noun (e.g., Data Science)
        if word1.pos_ == "NOUN" and word2.pos_ == "NOUN":
            patterns.add(f"{word1.text} {word2.text}")

    return list(patterns)

# ---------------------------------------
# Helper Function 3: Named Entity Recognition (NER)
# Extracts ORG, PRODUCT, and WORK_OF_ART as possible skills
# ---------------------------------------
def extract_entities(text):
    doc = nlp(text)
    skills = []

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:
            skills.append(ent.text)
    return list(set(skills))

# ---------------------------------------
# Question 5.1: Combine All Skill Extraction Methods
# ---------------------------------------
def extract_all_skills(resume_text):
    # Method 1: Keyword Matching
    matched_skills = extract_skills(resume_text, SKILL_DATABASE)

    # Combine all technical categories
    technical_skills = []
    for cat in ['programming_languages', 'frameworks', 'databases', 'cloud']:
        if cat in matched_skills:
            technical_skills.extend(matched_skills[cat])

    # Soft skills
    soft_skills = matched_skills.get('soft_skills', [])

    # Method 2: POS Pattern Matching
    pos_patterns = find_skill_patterns(resume_text)
    technical_skills.extend(pos_patterns)

    # Method 3: Named Entity Recognition
    ner_skills = extract_entities(resume_text)
    technical_skills.extend(ner_skills)

    # Deduplicate
    technical_skills = sorted(list(set(technical_skills)))
    soft_skills = sorted(list(set(soft_skills)))

    # Combine all
    all_skills = sorted(list(set(technical_skills + soft_skills)))

    return {
        'technical_skills': technical_skills,
        'soft_skills': soft_skills,
        'all_skills': all_skills
    }

# ---------------------------------------
# Question 5.2: Generate Skill Report
# ---------------------------------------
def generate_skill_report(skills_dict):
    tech = skills_dict['technical_skills']
    soft = skills_dict['soft_skills']
    total = len(skills_dict['all_skills'])

    print("=== SKILL EXTRACTION REPORT ===\n")

    print(f"TECHNICAL SKILLS ({len(tech)}):")
    for s in tech:
        print(f"  • {s}")

    print(f"\nSOFT SKILLS ({len(soft)}):")
    for s in soft:
        print(f"  • {s}")

    tech_percent = (len(tech) / total * 100) if total else 0
    soft_percent = (len(soft) / total * 100) if total else 0

    print(f"\nSUMMARY:")
    print(f"  Total Skills: {total}")
    print(f"  Technical: {len(tech)} ({tech_percent:.0f}%)")
    print(f"  Soft Skills: {len(soft)} ({soft_percent:.0f}%)")

# ---------------------------------------
# Test the Complete Skill Extractor
# ---------------------------------------
if __name__ == "__main__":
    resume = """
    SKILLS:
    Programming: Python, Java, JavaScript
    Frameworks: TensorFlow, React, Django
    Experience in Machine Learning and Deep Learning
    Strong analytical and problem-solving skills
    """

    skills = extract_all_skills(resume)
    generate_skill_report(skills)
