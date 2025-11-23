import re
from typing import Dict, List

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}")

SECTION_HEADERS = [
    "summary", "profile", "skills", "experience", "work experience",
    "education", "projects", "certifications", "contact", "about"
]

def clean_text(text: str) -> str:
    """
    Basic cleaning: normalize whitespace and remove weird control chars.
    """
    if not text:
        return ""

    txt = text.replace("\r\n", "\n").replace("\r", "\n")
    # remove non-printable characters
    txt = "".join(ch for ch in txt if ord(ch) >= 9 and ord(ch) <= 126 or ch == "\n")
    # collapse multiple newlines and spaces
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()

def extract_contact_info(text: str) -> Dict[str, List[str]]:
    """
    Return found emails and phone numbers (as lists).
    """
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    # basic normalization of phones (strip spaces)
    phones = [re.sub(r"[^\d\+]", "", p) for p in phones]
    return {
        "emails": list(set(emails)),
        "phone_numbers": list(set(phones))
    }

def split_sections(text: str) -> Dict[str, str]:
    """
    Heuristic section splitter:
    - Looks for lines that match known section headers.
    - Groups following lines under that header until next header.
    Returns dict header -> section text. If no header found, entire text under 'full_text'.
    """
    lines = [l.strip() for l in text.splitlines()]
    sections = {}
    current = "full_text"
    buffer = []

    def flush():
        if buffer:
            prev = sections.get(current, "")
            combined = (prev + "\n" + "\n".join(buffer)).strip() if prev else "\n".join(buffer).strip()
            sections[current] = combined
        
    for line in lines:
        low = line.lower().strip(": ")
        # treat short lines as potential headers if they match keywords
        if any(low == h for h in SECTION_HEADERS) or (len(line) < 40 and any(h in low for h in SECTION_HEADERS)):
            # new section
            flush()
            buffer = []
            # normalize header to the matched canonical header
            matched = next((h for h in SECTION_HEADERS if h in low), low)
            current = matched
        else:
            buffer.append(line)
    
    flush()
    # ensure at least full_text exists
    if not sections:
        sections["full_text"] = text
    return sections