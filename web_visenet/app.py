import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path
import hashlib
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import streamlit as st
import streamlit.components.v1 as components

from web_visenet.main_page import show_main_page

BASE_DIR = os.path.dirname(__file__)
USERS_PATH = os.path.join(BASE_DIR, "json", "users.json")
FLAG_PATH = os.path.join(BASE_DIR, "json", "login_flag.json")

# Function to hash password
def hash_password(password: str) -> str:
    """
    Hash the password using SHA-256
    
    Parameters:
        password: The plaintext password to hash
    
    Returns:
        str: The hexadecimal representation of the hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()

# Function to load users from file
def load_users() -> dict:
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

# Save users to file
def save_users(users: dict) -> None:
    """
    Save the users dictionary to a JSON file
    
    Parameters:
        users: A dictionary mapping usernames to hashed passwords
    """
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


_server_thread = None

# Open a local server to handle login/register
def start_local_server(port: int = 8765) -> None:
    global _server_thread
    if _server_thread is not None:
        return  

    class Handler(BaseHTTPRequestHandler):
        def _set_cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self) -> None:
            self.send_response(200)
            self._set_cors()
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/save_user":
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            typ = data.get("type")
            username = (data.get("username") or "").strip()
            password = (data.get("password") or "").strip()

            users = load_users()

            if typ == "register" and username and password:
                # New user registration
                users[username] = hash_password(password)
                save_users(users)
                with open(FLAG_PATH, "w", encoding="utf-8") as f:
                    json.dump({"username": username}, f)

            elif typ == "login" and username and password:
                # Check password
                stored = users.get(username)
                if stored and stored == hash_password(password):
                    with open(FLAG_PATH, "w", encoding="utf-8") as f:
                        json.dump({"username": username}, f)

            self.send_response(200)
            self._set_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

        def log_message(self, format, *args) -> None:
            return

    server = HTTPServer(("localhost", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _server_thread = thread


# Interface HTML for login/register
def load_auth_html() -> str:
    base_dir = Path(__file__).parent / "components" / "auth_component"
    html_path = base_dir / "auth.html"
    css_path = base_dir / "auth.css"
    js_path = base_dir / "auth.js"

    html = html_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")
    js = js_path.read_text(encoding="utf-8")

    # Replace link/script tags with inline content
    html = html.replace(
        '<link rel="stylesheet" href="auth.css">',
        f"<style>\n{css}\n</style>"
    )
    html = html.replace(
        '<script src="auth.js"></script>',
        f"<script>\n{js}\n</script>"
    )
    return html

def main() -> None:
    st.set_page_config(page_title="ViseNet", layout="wide")

    if "local_server_started" not in st.session_state:
        start_local_server(8765)
        st.session_state["local_server_started"] = True

    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user", "")

    if os.path.exists(FLAG_PATH):
        try:
            with open(FLAG_PATH, "r", encoding="utf-8") as f:
                flag = json.load(f)
                username = flag.get("username")
            os.remove(FLAG_PATH)
            if username:
                st.session_state["logged_in"] = True
                st.session_state["user"] = username
                st.rerun()
        except Exception:
            pass

    # If already logged in, show main page
    if st.session_state["logged_in"]:
        st.sidebar.success(f"Welcome, {st.session_state['user']} 👋")

        # Log out button
        if st.sidebar.button("Log out"):
            st.session_state["logged_in"] = False
            st.session_state["user"] = ""
            st.rerun()

        # Upload file (csv, json, txt)
        uploaded_file = st.sidebar.file_uploader(
            "Upload stock codes information",
            type=["csv", "json", "txt"]
        )

        if uploaded_file is not None:
            # Save file bytes to session
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()

        # Enter email to notify
        email_to_notify = st.sidebar.text_input("Enter the email you want to send notifications to:")

        if email_to_notify:
            st.sidebar.write(f"📧 Confirm to send notifications to: {email_to_notify}")

        # Show main page
        show_main_page()
        return


    # Login page
    components.html(load_auth_html(), height=600, scrolling=False)


if __name__ == "__main__":
    main()
