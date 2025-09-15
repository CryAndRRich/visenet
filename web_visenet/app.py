import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import json
import hashlib
import os
from web_visenet.main_page import show_main_page

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists("web_visenet/json/users.json"):
        with open("web_visenet/json/users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("web_visenet/json/users.json", "w") as f:
        json.dump(users, f)

# Hàm đăng ký
def register():
    st.subheader("Đăng ký tài khoản")
    username = st.text_input("Tên đăng nhập", key="reg_user")
    password = st.text_input("Mật khẩu", type="password", key="reg_pass")
    if st.button("Đăng ký"):
        if not username or not password:
            st.warning("Vui lòng điền đủ thông tin")
            return
        users = load_users()
        if username in users:
            st.error("Tên đăng nhập đã tồn tại!")
        else:
            users[username] = hash_password(password)
            save_users(users)
            st.success("Đăng ký thành công! Hãy đăng nhập ngay.")

# Hàm đăng nhập
def login():
    st.subheader("Đăng nhập")
    username = st.text_input("Tên đăng nhập", key="login_user")
    password = st.text_input("Mật khẩu", type="password", key="login_pass")
    if st.button("Đăng nhập"):
        users = load_users()
        if username in users and users[username] == hash_password(password):
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            # Reload để tự động vào main page
            st.session_state.rerun() 
        else:
            st.error("Sai tên đăng nhập hoặc mật khẩu")

# Hàm đăng xuất
def logout():
    st.session_state["logged_in"] = False
    st.session_state["user"] = ""
    st.session_state.rerun()  # reload về trang login/register

# Hàm chính
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user"] = ""

    if st.session_state["logged_in"]:
        st.sidebar.write(f"Xin chào, {st.session_state['user']}")
        if st.sidebar.button("Đăng xuất"):
            logout()
        
        # Hiển thị nội dung chính
        show_main_page()
    else:
        st.title("Chào mừng! Vui lòng đăng nhập hoặc đăng ký")
        choice = st.radio("Chọn hành động:", ["Đăng nhập", "Đăng ký"])
        if choice == "Đăng nhập":
            login()
        else:
            register()

if __name__ == "__main__":
    main()
