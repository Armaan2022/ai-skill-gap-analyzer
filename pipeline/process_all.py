import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from pipeline.resume_parser import parse_and_save_file
from pipeline.jd_parser import parse_and_save_jd

RESUME_IN = Path("data/resumes")
JD_IN = Path("data/job_descriptions")
OUT = Path("data/processed/json")

OUT.mkdir(parents=True, exist_ok=True)

def process_resumes():
    for p in RESUME_IN.glob("*.txt"):
        parse_and_save_file(str(p), str(OUT))
    
def process_jds():
    for p in JD_IN.glob("*.txt"):
        parse_and_save_jd(str(p), str(OUT))

if __name__ == "__main__":
    print("Processing resumes...")
    process_resumes()
    print("Processing job descriptions...")
    process_jds()
    print("Done. Processed JSON saved to data/processed/json")