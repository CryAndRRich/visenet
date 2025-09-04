## VISENET: Vietnamese Investment weighted-Scoring and Ensemble Network for Enhanced Trading
Đây là repository của mô hình **VISENET**, sản phẩm của đội thi **HD4K** trong **Vòng 02 Cuộc thi Data Science Talent Competition 2025**

```
visenet/
├── config/                                   # Đường dẫn thư mục
│   └── config.py/
│
├── data/                                     # File dữ liệu thu thập từ FiinQuantX
│
├── env/                                      # Thiết lập môi trường giao dịch
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
|   └── visenet_run_model.ipynb/              # File notebook chạy mô hình ensemble
|
├── report/                                   # Thư mục chứa file báo cáo pdf
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
- Chạy file `models/utils.py` tính toán và vẽ các biểu đồ để phân tích hiệu suất đầu tư

**Lưu ý**: 
- Đường dẫn file trong các file `.ipynb` và `.py` có thể không chính xác, nếu chạy hãy cẩn thận điều chỉnh lại
- Việc chạy các file notebook run_model sẽ tốn nhiều thời gian