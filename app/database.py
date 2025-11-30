from supabase import create_client, Client
from app.config import SUPABASE_URL, SUPABASE_KEY

# Initialize the client once and reuse it
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)