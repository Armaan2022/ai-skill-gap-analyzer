import os

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./skillgap.db")
CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]
