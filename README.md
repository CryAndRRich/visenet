## VISENET: Vietnamese Investment weighted-Scoring and Ensemble Network for Enhanced Trading
This is the repository of the **VISENET** model and the real-time alerting system, the winning solution developed by team **HD4K**, Champion of the **Data Science Talent Competition 2025**.

```
visenet/
├── config/                                   # Configuration directory
│   └── config.py/
│
├── data/                                     # Data files collected from FiinQuantX
│
├── env/                                      # Virtual trading environment setup
│   ├── EnvMultipleStock_trade.py/            
|   ├── EnvMultipleStock_train.py/
|   └── EnvMultipleStock_validation.py/
│
├── preprocess/
│   ├── get_data.py/                          # Function to fetch stock data from FiinQuantX
|   └── preprocessor.py/                      # Preprocessing functions, TA and FA indicators
│
├── models/
|   ├── wscoring.py/                          # Weighted-scoring model to select top 30 stocks
|   ├── ensemble.py/                          # Ensemble model combining A2C, PPO, TD3
|   └── utils.py/                             # Utility functions for performance calculation and visualization
| 
├── runDRL.py/                                # Function to run the ensemble model
|
├── web_visenet/                              # Dashboard interface
|   ├── .streamlit/                           
|   |   └── config.toml/                      # Configuration for web interface
|   |
|   ├── components/auth_component/            # Login page component
|   |   ├── auth.html/  
|   |   ├── auth.js/  
|   |   └── auth.css/  
|   |
|   ├── json/                                 
|   |   └── users.json/                       # File storing user account information
|   |
|   ├── send_notifications.py/                # Function to send email notifications
|   ├── main_page.py/                         # Main dashboard page displaying data
|   └── app.py/                               # Web entrypoint: login + main page navigation
|
├── backtesting/                              # Backtesting notebooks and results
|   ├── results/                              # Zipped results of ensemble model runs
|   |
|   ├── trained_models/                       # Zipped trained models: A2C, PPO, TD3
|   |
|   ├── wscoring.ipynb/                       # Notebook: scoring & top stock selection
|   |
|   ├── a2c_run_model.ipynb/                  # Notebook: A2C model run
|   ├── ppo_run_model.ipynb/                  # Notebook: PPO model run
|   ├── td3_run_model.ipynb/                  # Notebook: TD3 model run
|   ├── visenet_run_model.ipynb/              # Notebook: ensemble model run
|   |
|   ├── demo_realtime_alerts.ipynb/           # Notebook: ensemble model with real-time alerts
|   └── demo_visenet.mp4/                     # VISENET demo video
|
├── report/                                   # Reports and related files
|   ├── img/                                  # Figures used in reports
|   |
|   ├── problem2_HD4K.pdf/                    # VISENET report for Round 02
|   |
|   ├── problem3_HD4K.pdf/                    # VISENET 2.0 real-time alerting system report for Round 03
|   └── slide_problem3_HD4K(pdf).pdf/         # VISENET presentation slides (PDF)
|
├── LICENSE/
├── requirements.txt/                         # Required libraries
└── README.md
```

## How to Run
### 1. Clone the repository
```
git clone https://github.com/CryAndRRich/visenet.git
```
### 2. Install dependencies
```
pip install -r visenet/requirements.txt
```

### 3. Run data processing, stock filtering, and model training
- Use `data/get_data.py`, `data/preprocessor.py` to fetch and preprocess data
- Run `models/wscoring.py` to select top 30 stocks (see notebook `backtesting/wscoring.ipynb`)
- Run `runDRL.py` to train the ensemble model (see notebook `backtesting/visenet_run_model.ipynb`)
- Run `backtesting/demo_realtime_alerts.ipynb` to execute ensemble model with real-time alerts
- Use `models/utils.py` for performance calculations and visualization

**Note**: 
- File paths in `.ipynb` and `.py` scripts may not be exact, adjust accordingly if needed
- Running notebooks from scratch may take significant time

### 4. Run the web dashboard
```
streamlit run web_visnet/app.py
```