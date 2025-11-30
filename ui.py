import streamlit as st
import requests
from PIL import Image
import io

# --- CONFIGURATION ---
API_URL = "http://localhost:8000"
ADMIN_KEY = "my_secure_admin_password" # Must match API_SECRET in .env

st.set_page_config(page_title="God Tier Vision", page_icon="👁️", layout="wide")

# --- CUSTOM CSS (Cyberpunk Style) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #00cc00; 
        border: none;
        border-radius: 4px;
    }
    .success-box {
        padding: 20px;
        background-color: #1c2e21;
        border: 1px solid #00cc00;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("👁️ God Tier Product Recognizer")
st.caption(f"Connected to AI Brain at: {API_URL}")

tabs = st.tabs(["🔍 Live Search (The User)", "🛡️ Admin Panel (Add Products)"])

# --- TAB 1: SEARCH ---
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Upload User Photo")
        st.info("Upload a 'messy' photo (e.g., watch on wrist, shoe on foot).")
        search_file = st.file_uploader("Choose image...", type=['jpg', 'png', 'jpeg'], key="search")

        if search_file:
            st.image(search_file, caption="User Input", use_container_width=True)
            
            if st.button("Identify Product", use_container_width=True):
                with st.spinner("🤖 AI Processing: Cropping -> SigLIP -> DINOv2..."):
                    try:
                        files = {"file": search_file.getvalue()}
                        response = requests.post(f"{API_URL}/search", files=files)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if data.get("found"):
                                prod = data["product"]
                                # Store result in session state to display in col2
                                st.session_state['result'] = prod
                            else:
                                st.error(data.get("message", "No match found"))
                                st.session_state['result'] = None
                        else:
                            st.error(f"Server Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Failed. Is main.py running? {e}")

    with col2:
        st.header("AI Findings")
        if 'result' in st.session_state and st.session_state['result']:
            prod = st.session_state['result']
            
            # Display Result Card
            st.markdown(f"""
            <div class="success-box">
                <h2>✅ Match Found!</h2>
                <h1>{prod['name']}</h1>
                <h3>Price: {prod['price']}</h3>
                <p>AI Confidence Score: {prod['score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(prod['image'], caption="Database Reference Photo (Studio Shot)", width=400)
            
            with st.expander("See AI Logic"):
                st.write("The AI successfully cropped the user image, found conceptually similar items via SigLIP, and verified the texture using DINOv2.")

# --- TAB 2: ADMIN ---
with tabs[1]:
    st.header("Add New Inventory")
    st.warning("🔒 restricted access")
    
    with st.form("add_product"):
        name = st.text_input("Product Name")
        price = st.text_input("Price (e.g., $150)")
        file = st.file_uploader("Upload Studio Photo (Clean Background)", type=['jpg', 'png', 'jpeg'])
        
        # Secret Key Input (Mock Auth)
        key_input = st.text_input("Admin Key", type="password")
        
        submitted = st.form_submit_button("Upload to Database")
        
        if submitted:
            if not file or not name or not price:
                st.error("Please fill all fields.")
            elif key_input != ADMIN_KEY:
                st.error("Invalid Admin Key!")
            else:
                with st.spinner("Generating Dual-Vector Embeddings..."):
                    try:
                        files = {"file": file.getvalue()}
                        payload = {"name": name, "price": price}
                        headers = {"x-api-key": key_input}
                        
                        response = requests.post(
                            f"{API_URL}/add_product", 
                            data=payload, 
                            files=files, 
                            headers=headers
                        )
                        
                        if response.status_code == 200:
                            st.success(f"✅ Successfully added {name}!")
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")