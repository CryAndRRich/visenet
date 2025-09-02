import pandas as pd
import numpy as np
import time
import gym

# ================================================================
# Mô hình DRL
# ================================================================
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv


# ================================================================
# Môi trường giao dịch cổ phiếu
# ================================================================
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade
from config import config
from preprocess.data_split import data_split


# ================================================================
# Hàm huấn luyện DRL
# ================================================================
def train_A2C(env_train, model_name, timesteps=25000):
    """Train A2C model"""
    start = time.time()
    model = A2C("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (A2C): ", (end - start) / 60, " minutes")
    return model


def train_TD3(env_train, model_name, timesteps=10000):
    """Train TD3 model (thay cho DDPG)"""
    n_actions = env_train.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    start = time.time()
    model = TD3("MlpPolicy", env_train, action_noise=action_noise, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (TD3): ", (end - start) / 60, " minutes")
    return model


def train_PPO(env_train, model_name, timesteps=50000):
    """Train PPO model"""
    start = time.time()
    model = PPO("MlpPolicy", env_train, ent_coef=0.005, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (PPO): ", (end - start) / 60, " minutes")
    return model


# ================================================================
# Hàm dự đoán DRL
# ================================================================
def DRL_prediction(df, model, name, last_state, iter_num, unique_trade_date, rebalance_window, turbulence_threshold, initial):
    """Dự đoán dựa trên mô hình đã huấn luyện"""

    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _ = model.predict(obs_trade)
        obs_trade, _, _, _ = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.envs[0].state

    if isinstance(last_state, (list, np.ndarray)):
        df_last_state = pd.DataFrame([last_state]) 
    else:
        df_last_state = pd.DataFrame({"last_state": [last_state]}) 

    df_last_state.to_csv(f"results/last_state_{name}_{iter_num}.csv", index=False)

    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    for i in range(len(test_data.index.unique())):
        action, _ = model.predict(test_obs)
        test_obs, _, _, _ = test_env.step(action)


def get_validation_sharpe(iteration):
    """Tính toán Sharpe ratio từ kết quả xác nhận"""
    df_total_value = pd.read_csv(
        f"results/account_value_validation_{iteration}.csv", 
        index_col=0
    )
    df_total_value.columns = ["account_value_train"]
    df_total_value["daily_return"] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value["daily_return"].mean() / df_total_value["daily_return"].std()
    return sharpe

# ================================================================
# Chiến lược Ensemble
# ================================================================
def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window):
    """Chiến lược Ensemble kết hợp PPO, A2C và TD3"""
    print("============Start Ensemble Strategy============")
    last_state_ensemble = []

    ppo_sharpe_list = []
    td3_sharpe_list = []
    a2c_sharpe_list = []
    model_use = []

    insample_turbulence = df[(df.timestamp < 20240101) & (df.timestamp >= 20181129)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=["timestamp"])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")

        if i - rebalance_window - validation_window == 0:
            initial = True
        else:
            initial = False

        # set turbulence threshold
        end_date_index = df.index[df["timestamp"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        print(df.index[df["timestamp"] == unique_trade_date[i - rebalance_window - validation_window]].to_list())
        end_date_index = int(end_date_index)
        start_date_index = end_date_index - validation_window * 30 + 1
        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=["timestamp"])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            turbulence_threshold = insample_turbulence_threshold
        else:
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        # training env
        train = data_split(df, start=20090000,
                           end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        # validation env
        validation = data_split(df,
                                start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()

        # Train A2C
        print("======A2C Training========")
        model_a2c = train_A2C(env_train, f"A2C_30k_dow_{i}", timesteps=30000)
        DRL_validation(model_a2c, validation, env_val, obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print("A2C Sharpe Ratio: ", sharpe_a2c)

        # Train PPO
        print("======PPO Training========")
        model_ppo = train_PPO(env_train, f"PPO_100k_dow_{i}", timesteps=100000)
        DRL_validation(model_ppo, validation, env_val, obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        # Train TD3
        print("======TD3 Training========")
        model_td3 = train_TD3(env_train, f"TD3_10k_dow_{i}", timesteps=10000)
        DRL_validation(model_td3, validation, env_val, obs_val)
        sharpe_td3 = get_validation_sharpe(i)
        print("TD3 Sharpe Ratio: ", sharpe_td3)

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        td3_sharpe_list.append(sharpe_td3)

        # model selection
        if (sharpe_ppo >= sharpe_a2c) and (sharpe_ppo >= sharpe_td3):
            model_ensemble = model_ppo
            model_use.append("PPO")
        elif (sharpe_a2c > sharpe_ppo) and (sharpe_a2c > sharpe_td3):
            model_ensemble = model_a2c
            model_use.append("A2C")
        else:
            model_ensemble = model_td3
            model_use.append("TD3")

        # Trading
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        last_state_ensemble = DRL_prediction(df, model_ensemble, "ensemble",
                                             last_state_ensemble, i,
                                             unique_trade_date,
                                             rebalance_window,
                                             turbulence_threshold,
                                             initial)

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
