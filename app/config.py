import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_SECRET = os.getenv("API_SECRET")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ MISSING CONFIG: Please set SUPABASE_URL and SUPABASE_KEY in .env")