import io
import json
import torch
import asyncio
import uuid
import os
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse
from PIL import Image
from dotenv import load_dotenv

from app.database import supabase
from app.ai_engine import load_models, smart_crop, get_siglip_vector, get_dino_vector
from app.config import API_SECRET

# --- CONFIGURATION ---
load_dotenv()
FB_VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    print("✅ System Ready: AI + Facebook Bot Active.")
    yield

app = FastAPI(lifespan=lifespan)

# --- 1. FACEBOOK HELPER FUNCTIONS ---

def send_fb_message(recipient_id, text):
    """Sends a text reply back to the user on Messenger."""
    if not FB_PAGE_ACCESS_TOKEN:
        print("❌ Error: Missing FB_PAGE_ACCESS_TOKEN in .env")
        return

    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text}
    }
    try:
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            print(f"FB Send Error: {r.text}")
    except Exception as e:
        print(f"Connection Error: {e}")

def send_fb_image(recipient_id, image_url):
    """Sends an image attachment to the user on Messenger."""
    if not FB_PAGE_ACCESS_TOKEN: return

    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": "image", 
                "payload": {
                    "url": image_url, 
                    "is_reusable": True
                }
            }
        }
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"FB Image Error: {e}")

def handle_fb_image(sender_id, image_url):
    """Background Task: Downloads image -> Runs AI -> Sends Reply"""
    try:
        send_fb_message(sender_id, "👁️ Analyzing your image...")
        
        # 1. Download Image from Facebook
        img_resp = requests.get(image_url)
        if img_resp.status_code != 200:
            send_fb_message(sender_id, "❌ I couldn't download that image.")
            return

        image_bytes = img_resp.content
        
        # 2. Run AI (Using your stable process_search)
        result = process_search(image_bytes)
        
        # 3. Reply based on result
        if result['found']:
            prod = result['product']
            # Format the confidence score nicely
            conf = int(prod['score'] * 100)
            
            # Send Text with Disclaimer
            msg = (
                f"✅ MATCH FOUND!\n\n"
                f"Product: {prod['name']}\n"
                f"Price: {prod['price']}\n"
                f"Confidence: {conf}%\n\n"
                f"⚠️ Note: This result is automated by AI. Occasional errors may occur."
            )
            send_fb_message(sender_id, msg)
            
            # Send the exact DB Picture
            if prod.get('image'):
                send_fb_image(sender_id, prod['image'])
        else:
            send_fb_message(sender_id, "❌ I'm not sure what that is. Try a clearer photo.")
            
    except Exception as e:
        print(f"Error processing FB image: {e}")
        send_fb_message(sender_id, "⚠️ System Error. Please try again.")

# --- 2. FACEBOOK WEBHOOK ENDPOINTS ---

@app.get("/webhook")
async def verify_fb_webhook(request: Request):
    """
    Facebook Verification Handshake.
    Must return Plain Text to pass the test.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == FB_VERIFY_TOKEN:
        return PlainTextResponse(content=challenge, status_code=200)
    
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def handle_fb_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives messages/images from Users."""
    data = await request.json()
    
    # Check if this is a page event
    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for event in entry.get("messaging", []):
                if "sender" not in event: continue
                sender_id = event["sender"]["id"]
                
                # Check if message has an image attachment
                if "message" in event and "attachments" in event["message"]:
                    for att in event["message"]["attachments"]:
                        if att["type"] == "image":
                            image_url = att["payload"]["url"]
                            # Run AI in background so we don't block the webhook
                            background_tasks.add_task(handle_fb_image, sender_id, image_url)
                
                # Handle text messages
                elif "message" in event and "text" in event["message"]:
                    text = event["message"]["text"]
                    if "hi" in text.lower() or "hello" in text.lower():
                        send_fb_message(sender_id, "Hello! Send me a photo of a watch or product.")

        return {"status": "ok"}
    
    raise HTTPException(status_code=404)

# --- 3. CORE AI LOGIC (Your Stable Code) ---

@app.get("/")
def health():
    return {"status": "online", "mode": "Stable Production + FB Bot"}

@app.post("/search")
async def search_product(file: UploadFile = File(...)):
    # 1. Read Image
    content = await file.read()
    # Run in thread to prevent blocking
    result = await asyncio.to_thread(process_search, content)
    return result

def process_search(image_bytes):
    original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # STAGE 1: CROP
    clean_img = smart_crop(original_img)
    
    # STAGE 2: SEARCH (SigLIP)
    query_vec = get_siglip_vector(clean_img)
    
    try:
        response = supabase.rpc("match_products", {
            "query_embedding": query_vec,
            "match_threshold": 0.10, # Loose threshold
            "match_count": 15
        }).execute()
    except Exception as e:
        return {"found": False, "error": f"Database Error: {e}"}
        
    candidates = response.data
    if not candidates:
        return {"found": False, "message": "No match found."}

    # STAGE 3: VERIFY (DINOv2)
    user_dino_vec = torch.tensor(get_dino_vector(clean_img))
    
    scored_candidates = []
    
    for cand in candidates:
        # Parse Vector
        cand_vec_data = cand['dino_embedding']
        if isinstance(cand_vec_data, str):
            cand_vec_data = json.loads(cand_vec_data)
            
        cand_tensor = torch.tensor(cand_vec_data)
        
        # Calculate Similarity (0.0 to 1.0)
        score = torch.dot(user_dino_vec, cand_tensor).item()
        
        scored_candidates.append({
            "name": cand['name'],
            "price": cand['price'],
            "image": cand['image_url'],
            "score": score
        })
    
    # Sort: Highest Score First
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    best_match = scored_candidates[0]
    
    # STRICT THRESHOLD
    STRICT_THRESHOLD = 0.65
    
    if best_match['score'] < STRICT_THRESHOLD:
        return {
            "found": False,
            "message": "Product looks similar, but I cannot confirm the exact model.",
            "best_guess": best_match['name'],
            "confidence": round(best_match['score'], 2)
        }

    return {"found": True, "product": best_match}

@app.post("/add_product")
async def add_product(
    name: str = Form(...),
    price: str = Form(...),
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    
    # Unique Filename
    file_ext = file.filename.split('.')[-1]
    filename = f"{name.replace(' ', '')}_{str(uuid.uuid4())[:8]}.{file_ext}"
    
    supabase.storage.from_("product-images").upload(
        filename, 
        content, 
        {"content-type": file.content_type, "upsert": "true"}
    )
    public_url = supabase.storage.from_("product-images").get_public_url(filename)
    
    # Generate & Save
    supabase.table("products").insert({
        "name": name,
        "price": price,
        "image_url": public_url,
        "siglip_embedding": get_siglip_vector(image),
        "dino_embedding": get_dino_vector(image)
    }).execute()
    
    return {"status": "ok", "product": name}