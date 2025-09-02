import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()


EMBED_MODEL = "models/embedding-001"
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
COLLECTION_NAME = "sec_filings"
NOT_FOUND_MSG = """
Sorry, we couldn't find an answer based on the documents provided. Please try rephrasing your question or check if the information is available in the sources."""
GENERATIVE_MODEL = "gemini-2.5-flash"
TOP_K = 5
PREFETCH = 30
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
PDF_DIR = "pdf/"
CHROMA_DB_PATH = "chroma_db/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# generic SEC filing pattern regex
SEC_ITEM_PATTERNS = [
    r"Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
    r"ITEM\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
    r"Part\s+([IV]+)\s*[,\-]\s*Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
]


FINANCE_STOPWORD = {
    "revenue",
    "earnings",
    "profit",
    "loss",
    "cash",
    "debt",
    "assets",
    "liabilities",
    "equity",
    "shares",
    "dividend",
    "eps",
    "ebitda",
    "operating",
    "income",
    "expenses",
    "margin",
    "growth",
    "risk",
    "material",
    "adverse",
    "segment",
    "goodwill",
    "impairment",
    "depreciation",
    "amortization",
    "taxes",
    "automotive",
    "interest",
    "cost",
    "sales",
    "services",
    "products",
    "customers",
    "market",
    "competition",
    "regulatory",
    "compliance",
    "litigation",
    "contingencies",
    "apple",
    "tesla",
    "automotive",
    "technology",
    "manufacturing",
    "unresolved",
    "leasing",
}
