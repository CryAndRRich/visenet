import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Each time trading a maximum of 100 shares
HMAX_NORMALIZE = 100
# Account balance
INITIAL_ACCOUNT_BALANCE = 1000000
# Number of stocks in our portfolio
STOCK_DIM = 30
# Transaction fee: 0.1% commission
TRANSACTION_FEE_PERCENT = 0.001

# Turbulence threshold: 140 (suggested by Dantas et al. (2020), can be adjusted according to the user's needs)
# TURBULENCE_THRESHOLD = 140
REWARD_SCALING = 1e-4

class StockEnvValidation(gym.Env):
    """Stock Trading Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, 
                 df: pd.DataFrame, 
                 day: int = 0, 
                 turbulence_threshold: int = 140, 
                 iteration = "") -> None:
        # super(StockEnv, self).__init__()
        
        # money = 10
        # scope = 1
        self.day = day
        self.df = df

        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,)) 

        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (181,))

        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold

        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.close.values.tolist() + \
                      [0] * STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()

        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0

        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        # self.reset()
        self._seed()
        
        self.iteration=iteration


    def _sell_stock(self, 
                    index: int, 
                    action: int) -> None:
        # Sell based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            if self.state[index + STOCK_DIM + 1] > 0:
                # Update balance
                self.state[0] += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * (1 - TRANSACTION_FEE_PERCENT)
                
                self.state[index + STOCK_DIM + 1] -= min(abs(action), self.state[index + STOCK_DIM + 1])
                self.cost +=self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * TRANSACTION_FEE_PERCENT
                self.trades += 1
            else:
                pass
        else:
            # If the turbulence exceeds the threshold, liquidate all positions
            if self.state[index + STOCK_DIM + 1] > 0:
                # Update balance
                self.state[0] += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * (1 - TRANSACTION_FEE_PERCENT)
                self.state[index + STOCK_DIM + 1] = 0
                self.cost += self.state[index + 1] * self.state[index + STOCK_DIM + 1] *  \
                              TRANSACTION_FEE_PERCENT
                self.trades += 1
            else:
                pass
    
    def _buy_stock(self, index, action):
        # Buy based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index + 1]
            # print("available_amount: {}".format(available_amount))
            
            # Update balance
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)

            self.state[index + STOCK_DIM + 1] += min(available_amount, action)
            
            self.cost+=self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            # If the turbulence exceeds the threshold, do not buy any stocks
            pass
        
    def step(self, actions: np.ndarray) -> tuple:
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory, "r")
            plt.savefig("results/account_value_validation_{}.png".format(self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv("results/account_value_validation_{}.csv".format(self.iteration))
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            # print("previous_total_asset: {}".format(self.asset_memory[0]))           

            # print("end_total_asset: {}".format(end_total_asset))
            # print("total_reward: {}".format(self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):61])) - self.asset_memory[0] ))
            # print("total_cost: ", self.cost)
            # print("total trades: ", self.trades)

            df_total_value.columns = ["account_value"]
            df_total_value["daily_return"] = df_total_value.pct_change(1)
            
            # df_rewards = pd.DataFrame(self.rewards_memory)
            # df_rewards.to_csv("results/account_rewards_trade_{}.csv".format(self.iteration))
            
            # print("total asset: {}".format(self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:]))))
            # with open("obs.pkl", "wb") as f:  
            #    pickle.dump(self.state, f)
            
            return self.state, self.reward, self.terminal,{}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * HMAX_NORMALIZE
            # actions = (actions.astype(int))
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)
            begin_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            # print("begin_total_asset: {}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print("take sell action".format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print("take buy action: {}".format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.turbulence = self.data["turbulence"].values[0]
            # print(self.turbulence)
            # print("stock_shares: {}".format(self.state[29:]))
            self.state =  [self.state[0]] + \
                           self.data.close.values.tolist() + \
                           list(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]) + \
                           self.data.macd.values.tolist() + \
                           self.data.rsi.values.tolist() + \
                           self.data.cci.values.tolist() + \
                           self.data.adx.values.tolist()
            
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            self.asset_memory.append(end_total_asset)
            # print("end_total_asset: {}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward: {}".format(self.reward))
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self) -> list:  
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        # self.iteration = self.iteration
        self.rewards_memory = []

        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.close.values.tolist() + \
                      [0] * STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist()  + \
                      self.data.cci.values.tolist()  + \
                      self.data.adx.values.tolist() 
            
        return self.state
    
    def render(self) -> list:
        return self.state
    
    def _seed(self, seed=None) -> list:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def save_asset_memory(self) -> list:
        return self.asset_memory
