
import pandas as pd
import numpy as np
import math
from itertools import compress

class Simulator:
    class OB:
        def __init__(self, current_time, bids, asks):
            self.current_time = current_time
            self.bids = bids
            self.asks = asks
            self.current_mid = (self.bids[0][1] + self.asks[0][1]) / 2
            self.best_bid = self.bids[0][1]
            self.best_ask = self.asks[0][1]
            
        def process_market_order(self, market_order):
            market_order = list(market_order)
            market_order.append(0.0)
            market_order.append(0.0)
            if (market_order[1] == 1.0):
                while (market_order[2] != 0.0):
                    if (len(self.asks) == 0.0): break
                    if (self.asks[0][1] >= market_order[2]):

                        p1 = market_order[4]
                        q1 = market_order[3]
                        p2 = self.asks[0][0]
                        q2 = market_order[2]
                        self.asks[0][1] -= q2
                        market_order[3] += q2                                   #Executed amount
                        market_order[4] = (p1 * q1 + p2 * q2) / (q1 + q2)       #VWAP
                        market_order[2] = 0.0                                   #Unexecuted amount
                        
                    elif (self.asks[0][1] < market_order[2]):

                        p1 = market_order[4]
                        q1 = market_order[3]
                        p2 = self.asks[0][0]
                        q2 = self.asks[0][1]
                        self.asks[0][1] = 0.0
                        market_order[3] += q2                                   #Executed amount
                        market_order[4] = (p1 * q1 + p2 * q2) / (q1 + q2)       #VWAP
                        market_order[2] -= q2                                   #Unexecuted amount
                        self.asks.pop(0)
                        
            elif (market_order[1] == -1.0):
                while (market_order[2] != 0.0  ):
                    if (len(self.bids) == 0): break
                    if (self.bids[0][1] >= market_order[2]):
 
                        p1 = market_order[4]
                        q1 = market_order[3]
                        p2 = self.bids[0][0]
                        q2 = market_order[2]
                        self.bids[0][1] -= q2
                        market_order[3] += q2                                   #Executed amount
                        market_order[4] = (p1 * q1 + p2 * q2) / (q1 + q2)       #VWAP
                        market_order[2] = 0.0                                   #Unexecuted amount
                    elif (self.bids[0][1] < market_order[2]):

                        p1 = market_order[4]
                        q1 = market_order[3]
                        p2 = self.bids[0][0]
                        q2 = self.bids[0][1]
                        self.asks[0][1] = 0.0
                        market_order[3] += q2                                   #Executed amount
                        market_order[4] = (p1 * q1 + p2 * q2) / (q1 + q2)       #VWAP
                        market_order[2] -= q2                                   #Unexecuted amount
                        self.bids.pop(0)
            if(len(self.bids)): self.best_bid = self.bids[0][1] 
            else: self.best_bid = 0.0
            if(len(self.asks)): self.best_ask = self.asks[0][1] 
            else: self.best_ask = 0.0

            return market_order
        
        def process_cancelation(self, cncl_order):
            if (cncl_order[1] == -1):
                 
                index = np.where([i == cncl_order[3] for i in [row[2] for row in self.asks]])[0]
                if(len(index)) :
                    self.asks[index[0]][1] = max(0.0, self.asks[index[0]][1] - cncl_order[2])
                else: return
                    
                    
            elif(cncl_order[1] == 1):
                
                index = np.where([i == cncl_order[3] for i in [row[2] for row in self.bids]])[0]
                if(len(index)) :
                    self.bids[index[0]][1] = max(0.0, self.bids[index[0]][1] - cncl_order[2])
                else: return
                
        def process_limit_order(self, limit_order):
            #update index in market order and cancelation to proceed
                
            
    def __init__(self, time, lambda_0_market, a, var_e, q_0, var_q, b, start_mid, depth, step, density, lambda_0_limit, lambda_0_cncl):
        self.time = time         # time horizon for which the trading occurs
        self.lambda_0_market = lambda_0_market # starting lambda - average number of orders per second
        self.a = a               # mean reversion coefficient for lambda
        self.var_e = var_e       # variance of error for lambda
        self.q_0 = q_0           # average deal quantity
        self.var_q = var_q       # variance of corresponding normal for deal quantity 
        self.b = b               # coefficient for direction correlation
        
        self.start_mid = start_mid # start mid
        self.depth = depth         # depth of the order book
        self.step = step           # step size of the price
        self.density = density     # density of the order book
        
        self.lambda_0_limit = lambda_0_limit
        self.lambda_0_cncl = lambda_0_cncl
        
        self.OB_0 = self.start_OB()
        
        self.market_orders_report = []
        self.market_orders = self.simulate_market_orders()
        self.limit_orders = self.simulate_limit_orders(self.lambda_0_limit)
        self.cancelations = self.simulate_limit_orders(self.lambda_0_cncl)
        
        
    def start_OB(self):
        lvls_bids = np.random.choice(self.depth, round(self.depth * self.density) , replace=False)
        lvls_bids.sort()
        lvls_asks = np.random.choice(self.depth, round(self.depth * self.density) , replace=False)
        lvls_asks.sort()
        bids = [[self.start_mid - (1 + i) * self.step, 
                      round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)), 4), i] 
                     for i in lvls_bids]
        asks = [[self.start_mid + (1 + i) * self.step, 
                      round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)),4), i]
                     for i in lvls_asks]
        ob = self.OB(0, bids, asks)
        return ob
     
    def simulate_market_orders(self):
        lambdas_market = []
        lambdas_market.append(self.lambda_0_market)
        times = []
        times.append(0)
        x_list = []
        quantities = []
        x_list.append(0)
        market_orders = []
         
        while (True):
            new_time = np.random.exponential(1 / lambdas_market[-1])
            if (new_time + times[-1] > self.time): break
                # recall that mean of lognormal is exp(mu + sigma^2/2)
            quantities.append(round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)), 4)) 
            times.append(new_time + times[-1])
            x_list.append(self.b * x_list[-1] + np.random.normal(0, 1))
            lambdas_market.append(abs(lambdas_market[-1] + self.a * (self.lambda_0_market - lambdas_market[-1]) + 
                                    np.random.normal(0, math.sqrt(self.var_e))))
        times.pop(0)
        x_list.pop(0)
        directions = np.sign(x_list)
        market_orders = list(zip(times, directions, quantities))
        return market_orders
    
    def simulate_limit_orders(self, lambda_):
        bid_lambdas_limit = []
        bid_lambdas_limit.append(lambda_)
        bid_times = []
        bid_times.append(0)
        bid_lvl_num_list = []
        bid_quantities = []
        
        ask_lambdas_limit = []
        ask_lambdas_limit.append(lambda_)
        ask_times = []
        ask_times.append(0)
        ask_lvl_num_list = []
        ask_quantities = []
        
        limit_orders = []
        
        while (True):
            new_time = np.random.exponential(1 / bid_lambdas_limit[-1])
            if (new_time + bid_times[-1] > self.time): break

            bid_quantities.append(round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)),4)) 
            bid_times.append(new_time + bid_times[-1])
            bid_lvl_num_list.append(np.random.choice(self.depth, replace=False))
            
            bid_lambdas_limit.append(abs(bid_lambdas_limit[-1] + self.a * (lambda_ - bid_lambdas_limit[-1]) + 
                                    np.random.normal(0, math.sqrt(self.var_e))))
            
        while (True):
            new_time = np.random.exponential(1 / ask_lambdas_limit[-1])
            if (new_time + ask_times[-1] > self.time): break

            ask_quantities.append(round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)),4)) 
            ask_times.append(new_time + ask_times[-1])
            ask_lvl_num_list.append(np.random.choice(self.depth, replace=False))
            
            ask_lambdas_limit.append(abs(ask_lambdas_limit[-1] + self.a * (lambda_ - ask_lambdas_limit[-1]) + 
                                    np.random.normal(0, math.sqrt(self.var_e))))
        
        bid_times.pop(0)
        ask_times.pop(0)
        limit_orders = list(zip(bid_times, np.full(len(bid_times), 1), bid_quantities, bid_lvl_num_list))
        limit_orders = limit_orders + list(zip(ask_times, np.full(len(ask_times), -1), ask_quantities, ask_lvl_num_list))
        limit_orders.sort()
        return limit_orders

    def matching(self):
        market_index = 0
        cncl_index = 0
        limit_index = 0
        len_m = len(self.market_orders)
        len_c = len(self.cancelations)
        len_l = len(self.limit_orders)
        self.market_orders.append((self.time + 1, -1.0, 1))   #breakpoint row,  will be deleted
        self.cancelations.append((self.time + 1, -1.0, 1, 0)) #breakpoint row,  will be deleted
        self.limit_orders.append((self.time + 1, -1.0, 1, 0)) #breakpoint row,  will be deleted
        
        
        order_books = []
        order_books.append(zip(0, self.OB_0))
        while (market_index != len_m & cncl_index != len_c & limit_index != len_l ):
            if ( self.market_orders[market_index][0] <= self.cancelations[cncl_index][0] 
            & self.market_orders[limit_index][0] <= self.limit_orders[limit_index][0] ):
                
                cur_ob = self.OB(self.market_orders[market_index][0], order_books[-1][1].bids, order_books[-1][1].asks)
                self.market_orders_report.append(cur_ob.process_market_order(self.market_orders[market_index]))
                order_books.append(zip(self.market_orders[market_index][0], cur_ob))
                market_index += 1
                
            elif( self.cancelations[cncl_index][0] <=  self.market_orders[market_index][0]
            & self.cancelations[cncl_index][0] <= self.limit_orders[limit_index][0] ):
                cur_ob = self.OB(self.cancelations[cncl_index][0], order_books[-1][1].bids, order_books[-1][1].asks)
                cur_ob.process_cancelation(self.cancelations[cncl_index])
                
time = 100
lambda_0 = 0.25
a = 0.05
var_e = 0.1
q_0 = 10
var_q = 2
b = 0.25
start_mid = 100
depth = 40
step = 0.0025
density = 0.8
lambda_0_limit = 5
lambda_0_cncl = 1

sim = Simulator(time, lambda_0, a, var_e, q_0, var_q, b, start_mid, depth, step, density, lambda_0_limit, lambda_0_cncl)
print(sim.OB_0.bids)
m = sim.OB_0.process_limit_order((0.11119819966925046, 1, 0.6831, 39))
print(sim.OB_0.bids)
