## VISENET: Vietnamese Investment weighted-Scoring and Ensemble Network for Enhanced Trading
Đây là repository của mô hình **VISENET** và hệ thống cảnh bảo thời gian thực, sản phẩm của đội thi **HD4K** ở **Vòng 02** và **Vòng 03** trong khuôn khổ **Cuộc thi Data Science Talent Competition 2025**

```
visenet/
├── config/                                   # Đường dẫn thư mục
│   └── config.py/
│
├── data/                                     # File dữ liệu thu thập từ FiinQuantX
│
├── env/                                      # Thiết lập môi trường ảo phục vụ giao dịch
│   ├── EnvMultipleStock_trade.py/            
|   ├── EnvMultipleStock_train.py/
|   └── EnvMultipleStock_validation.py/
│
├── preprocess/
│   ├── get_data.py/                          # Hàm lấy dữ liệu các mã cổ phiếu trên FiinQuantX
|   └── preprocessor.py/                      # Các hàm tiền xử lý, tính toán chỉ số TA, FA
│
├── models/
|   ├── wscoring.py/                          # Mô hình weighted-scoring chọn ra top 30 cổ phiếu
|   ├── ensemble.py/                          # Mô hình ensemble 3 thuật toán A2C, PPO, TD3
|   └── utils.py/                             # Các hàm tính toán và vẽ biểu đồ hiệu suất đầu tư
|
├── runDRL.py/                                # Hàm chạy mô hình ensemble
|
├── web_visenet/                              # Thư mục chứa giao diện dashboard
|   ├── .streamlit/                           
|   |   └── config.toml/                      # Lưu trữ các thông số cho giao diện web
|   |
|   ├── components/auth_component/            # Thư mục chứa file xây dựng login page
|   |   ├── auth.html/  
|   |   ├── auth.js/  
|   |   └── auth.css/  
|   |
|   ├── json/                                 
|   |   └── users.json/                       # File lưu trữ thông tin tài khoản người dùng
|   |
|   ├── send_notifications.py/                # Hàm gửi thông báo đến email người dùng
|   ├── main_page.py/                         # Trang chính dashboard hiển thị dữ liệu
|   └── app.py/                               # File chạy web đăng nhập và chuyển sang trang chính
|
├── backtesting/                              # Thư mục chứa notebook và kết quả chạy thử
|   ├── results/                              # File zip kết quả chạy mô hình ensemble
|   |
|   ├── trained_models/                       # File zip lưu mô hình đã huấn luyện A2C, PPO, TD3
|   |
|   ├── wscoring.ipynb/                       # File notebook chạy chọn trọng số và lọc top cổ phiếu
|   |
|   ├── a2c_run_model.ipynb/                  # File notebook chạy mô hình A2C
|   ├── ppo_run_model.ipynb/                  # File notebook chạy mô hình PPO
|   ├── td3_run_model.ipynb/                  # File notebook chạy mô hình TD3
|   ├── visenet_run_model.ipynb/              # File notebook chạy mô hình ensemble
|   |
|   ├── demo_realtime_alerts.ipynb/           # File notebook chạy mô hình kèm cảnh báo realtime
|   └── demo_visenet.mp4/                     # File demo giao diện visenet
|
├── report/                                   # Thư mục chứa file báo cáo pdf và các file liên quan
|   ├── img/                                  # Thư mục chứa ảnh sử dụng trong báo cáo
|   |
|   ├── problem2_HD4K.pdf/                    # File báo cáo mô hình VISENET vòng 02
|   |
|   ├── problem3_HD4K.pdf/                    # File báo cáo hệ thống cảnh báo VISENET 2.0 vòng 03
|   └── slide_problem3_HD4K(pdf).pdf/         # Slide thuyết trình VISENET (bản PDF)
|
├── LICENSE/
├── requirements.txt/                         # Cấu hình các thư viện cần thiết
└── README.md
```

## Hướng dẫn chạy
### 1. Tiến hành clone repository
```
git clone https://github.com/CryAndRRich/visenet.git
```
### 2. Cài đặt các thư viện cần thiết
```
pip install -r visenet/requirements.txt
```

### 3. Chạy các file đọc dữ liệu, lọc cổ phiếu và huấn luyện mô hình
- Các file `data/get_data.py`, `data/preprocessor.py` để lấy dữ liệu và tiền xử lý
- Chạy file `models/wscoring.py` để lọc chọn top 30 cổ phiếu (notebook `backtesting/wscoring.ipynb`)
- Chạy file `runDRL.py` để tiến hành huấn luyện mô hình ensemble (notebook `backtesting/visenet_run_model.ipynb`)
- Chạy notebook `backtesting/demo_realtime_alerts.ipynb` để chạy mô hình ensemble kèm cảnh báo realtime
- Chạy file `models/utils.py` tính toán và vẽ các biểu đồ để phân tích hiệu suất đầu tư

**Lưu ý**: 
- Đường dẫn file trong các file `.ipynb` và `.py` có thể không chính xác, nếu chạy hãy cẩn thận điều chỉnh lại
- Việc chạy các file notebook từ đầu sẽ tốn nhiều thời gian

### 4. Chạy và hiển thị web hệ thống cảnh báo
```
streamlit run web_visnet/app.py
```