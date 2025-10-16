Name: Priya Poosaala  
Email:priyapoosaala@gmail.com
rollno :26
college:SR University

It includes text preprocessing, POS tagging, skill database matching, named entity recognition,  
and multi-method skill extraction with reporting and bonus utilities.

-----------------------------------------------
TASKS COMPLETED
-----------------------------------------------
Task 1: Text Preprocessing (20 points)
   - Basic text cleaning
   - Tokenization using spaCy
   - Stop word removal (preserving programming languages)
   - Lemmatization
Task 2: POS Tagging (20 points)
   - POS tagging using spaCy
   - Extracting nouns and proper nouns as skills
   - Detecting Adjective + Noun skill patterns
Task 3: Skill Extraction (25 points)
   - Created SKILL_DATABASE with programming languages, frameworks, databases, cloud, and soft skills
   - Skill matching from resume text
   - Abbreviation normalization (e.g., ML → Machine Learning)
Task 4: Named Entity Recognition (15 points)
   - Extracted entities of types PERSON, ORG, GPE, PRODUCT
   - Manually annotated resume data for skill entities
Task 5: Complete Skill Extractor (20 points)
   - Combined keyword, POS pattern, and NER methods to extract skills
   - Generated formatted skill report with counts and percentages
Bonus Tasks (10 extra points)
   - Frequency counter for skills
   - Skill context extractor (sentence-level context around a skill)

-----------------------------------------------
PROBLEMS FACED
-----------------------------------------------
1. Installing and loading spaCy model (`en_core_web_sm`) initially took time due to package dependencies.
2. Handling punctuation in text cleaning while keeping (+, #, -, .) required fine-tuning of regex.
3. Multi-word skill extraction (e.g., “Machine Learning”) needed careful handling in both POS and NER phases.
4. Ensuring case-insensitive matching for all skills across methods.

-----------------------------------------------
TIME SPENT ON EACH TASK
-----------------------------------------------
Task 1 (Text Preprocessing): 1 hour  
Task 2 (POS Tagging): 1 hour  
Task 3 (Skill Extraction): 1.5 hours  
Task 4 (Named Entity Recognition): 1 hour  
Task 5 (Complete Skill Extractor): 1.5 hours  
Bonus Tasks: 0.5 hour  

-----------------------------------------------
TOTAL TIME SPENT: 6.5 hours
-----------------------------------------------

-----------------------------------------------
TOOLS AND LIBRARIES USED
-----------------------------------------------
- Python 3.10+
- spaCy (Natural Language Processing)
- re (Regular Expressions)
- collections.Counter
- Standard Python libraries

