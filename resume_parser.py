!pip install spacy pdfminer.six nltk
!python -m spacy download en_core_web_sm


from google.colab import files
uploaded = files.upload()
file_path = list(uploaded.keys())[0]


import re
import spacy
import nltk
from pdfminer.high_level import extract_text

nltk.download('punkt')
nltk.download('punkt_tab') # Added to download the missing resource

nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Extract Text
# -----------------------------
def get_text(pdf_path):
    return extract_text(pdf_path)

# -----------------------------
# Email
# -----------------------------
def get_email(text):
    emails = re.findall(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", text)
    return emails[0] if emails else None

# -----------------------------
# Phone
# -----------------------------
def get_phone(text):
    phones = re.findall(r"\+?\d[\d\s\-]{8,15}\d", text)
    return phones[0] if phones else None

# -----------------------------
# Name (NER)
# -----------------------------
def get_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

# -----------------------------
# Skills (Improved)
# -----------------------------
SKILLS_DB = [
    "python","java","c++","sql","machine learning","deep learning",
    "nlp","data analysis","tensorflow","pytorch","excel","power bi"
]

def get_skills(text):
    tokens = nltk.word_tokenize(text.lower())
    skills = []
    for skill in SKILLS_DB:
        if skill in text.lower():
            skills.append(skill)
    return list(set(skills))

# -----------------------------
# Education
# -----------------------------
def get_education(text):
    edu_keywords = [
        "bachelor","master","b.tech","m.tech","phd","bsc","msc"
    ]
    return [k for k in edu_keywords if k in text.lower()]

# -----------------------------
# Experience (Years Detection)
# -----------------------------
def get_experience(text):
    exp = re.findall(r"(\d+)\+?\s+years", text.lower())
    return max(exp) + " years" if exp else "Not found"

# -----------------------------
# Main Parser
# -----------------------------
def parse_resume(pdf_path):
    text = get_text(pdf_path)

    data = {
        "Name": get_name(text),
        "Email": get_email(text),
        "Phone": get_phone(text),
        "Skills": get_skills(text),
        "Education": get_education(text),
        "Experience": get_experience(text)
    }

    return data

# -----------------------------
# Run
# -----------------------------
result = parse_resume(file_path)

print("\n===== Parsed Resume =====")
for k, v in result.items():
    print(f"{k}: {v}")
