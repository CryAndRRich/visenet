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

# Hàm băm mật khẩu
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Hàm tải và lưu user
def load_users():
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

# Lưu user vào file
def save_users(users):
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


_server_thread = None

# Mở server local để xử lý đăng nhập/đăng ký
def start_local_server(port=8765):
    global _server_thread
    if _server_thread is not None:
        return  

    class Handler(BaseHTTPRequestHandler):
        def _set_cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(200)
            self._set_cors()
            self.end_headers()

        def do_POST(self):
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
                # thêm user mới
                users[username] = hash_password(password)
                save_users(users)
                # coi như đăng nhập luôn
                with open(FLAG_PATH, "w", encoding="utf-8") as f:
                    json.dump({"username": username}, f)

            elif typ == "login" and username and password:
                # kiểm tra mật khẩu
                stored = users.get(username)
                if stored and stored == hash_password(password):
                    with open(FLAG_PATH, "w", encoding="utf-8") as f:
                        json.dump({"username": username}, f)

            self.send_response(200)
            self._set_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

        def log_message(self, format, *args):
            return

    server = HTTPServer(("localhost", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _server_thread = thread


# Interface HTML cho đăng nhập/đăng ký
def load_auth_html():
    base_dir = Path(__file__).parent / "components" / "auth_component"
    html_path = base_dir / "auth.html"
    css_path = base_dir / "auth.css"
    js_path = base_dir / "auth.js"

    html = html_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")
    js = js_path.read_text(encoding="utf-8")

    # Thay thế thẻ link/script bằng nội dung inline
    html = html.replace(
        '<link rel="stylesheet" href="auth.css">',
        f"<style>\n{css}\n</style>"
    )
    html = html.replace(
        '<script src="auth.js"></script>',
        f"<script>\n{js}\n</script>"
    )
    return html

# Hàm chính
def main():
    st.set_page_config(page_title="ViseNet", layout="wide")

    if "local_server_started" not in st.session_state:
        start_local_server(8765)
        st.session_state["local_server_started"] = True

    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user", "")

    # đã login thì hiện page chính
    if st.session_state["logged_in"]:
        st.sidebar.success(f"Xin chào, {st.session_state['user']} 👋")

        # Nút đăng xuất
        if st.sidebar.button("Đăng xuất"):
            st.session_state["logged_in"] = False
            st.session_state["user"] = ""
            st.rerun()

        # Upload file (csv, json, txt)
        uploaded_file = st.sidebar.file_uploader(
            "Tải lên thông tin các mã cổ phiếu",
            type=["csv", "json", "txt"]
        )

        if uploaded_file is not None:
            # Lưu file bytes vào session
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()

        # Nhập email để gửi thông báo
        email_to_notify = st.sidebar.text_input("Nhập email bạn muốn gửi thông báo tới:")
        if email_to_notify:
            st.sidebar.write(f"📧 Xác nhận sẽ gửi thông báo tới: {email_to_notify}")

        # Hiển thị trang chính
        show_main_page()
        return


    # Login page
    components.html(load_auth_html(), height=600, scrolling=False)

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


if __name__ == "__main__":
    main()
