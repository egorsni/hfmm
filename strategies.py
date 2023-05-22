from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class BestPosStrategy:
    """
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(self, delay: float, hold_time: Optional[float] = None) -> None:
        """
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        """
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').delta)
        self.hold_time = hold_time

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Order]): list of all placed orders
        """

        # market data list
        md_list: List[MdUpdate] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            # get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else:
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                # place order
                bid_order = sim.place_order(receive_ts, 0.001, 'BID', best_bid)
                ask_order = sim.place_order(receive_ts, 0.001, 'ASK', best_ask)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, all_orders


class StoikovStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None, trade_size:Optional[float] = 0.01, risk_aversion:Optional[float] = 0.5) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        self.order_size = trade_size
        self.last_mid_prices = []
        self.asset_position = 0
        self.gamma = risk_aversion
        self.current_bid_order_id = None
        self.current_ask_order_id = None
        self.previous_bid_order_id = None
        self.previous_ask_order_id = None
        self.trades_dict = {'place_ts' :[], 'exchange_ts': [], 'receive_ts': [], 'trade_id': [],'order_id': [],'side': [], 'size': [], 'price': [],'execute':[], 'mid_price':[]}
        
    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    mid_price = (best_bid + best_ask)/2
                    if len(self.last_mid_prices) < 500:
                        self.last_mid_prices.append(mid_price)
                    else:
                        self.last_mid_prices.append(mid_price)
                        self.last_mid_prices.pop(0)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    self.trades_dict['place_ts'].append(update.place_ts)
                    self.trades_dict['exchange_ts'].append(update.exchange_ts)
                    self.trades_dict['receive_ts'].append(update.receive_ts)
                    self.trades_dict['trade_id'].append(update.trade_id)
                    self.trades_dict['order_id'].append(update.order_id)
                    self.trades_dict['side'].append(update.side)
                    self.trades_dict['size'].append(update.size)
                    self.trades_dict['price'].append(update.price)
                    self.trades_dict['execute'].append(update.execute)
                    self.trades_dict['mid_price'].append(mid_price)
                    trades_list.append(update)
                    if update.side == "ASK":
                        self.asset_position -= update.size
                    elif update.side == "BID":
                        self.asset_position += update.size
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                #place order
                '''
                reservation_price = s - q * gamma * (sigma**2) * (T - t)
                delta_bid and delta_ask are equivalently distant from the reservation_orice
                delta_bid + delta_ask = gamma * (sigma**2) * (T-t) + 2/gamma * ln(1 + gamma/k)
                k = K*alpha
                
                s      : current mid_price
                q      : current position in asset
                sigma  : parameter in the Geometric Brownian Motion equation [dS_t = sigma dw_t]
                T      : termination time
                t      : current time
                gamma  : risk-aversion parameter of the optimizing agents (across the economy)
                K      : higher K means that market order volumes have higher impact on best price changes
                alpha  : higher alpha means higher probability fo large market orders
                
                '''
                if len(self.last_mid_prices)==500:
                    sigma = np.std(self.last_mid_prices)## per update --> need to scale it to the "per second" terminology
                else:
                    sigma = 1
                sigma = sigma*np.sqrt(1/0.032)
                delta_t = 0.032 ## there is approximately 0.032 seconds in between the orderbook uprates (nanoseconds / 1e9 = seconds)
                k = 1.5
                q = self.asset_position
                ## mid_price = (best_bid + best_ask)/2 ## was defined previously
                reservation_price = mid_price - q*self.gamma*(sigma**2)*delta_t
                deltas_ = self.gamma * (sigma**2) * delta_t + 2/self.gamma * np.log(1 + self.gamma/k)
                bid_price = np.round(reservation_price - deltas_/2, 1)
                ask_price = np.round(reservation_price + deltas_/2, 1)
                
                bid_order = sim.place_order( receive_ts, self.order_size, 'BID', bid_price)
                ask_order = sim.place_order( receive_ts, self.order_size, 'ASK', ask_price)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order
                
                self.previous_bid_order_id = self.current_bid_order_id
                self.previous_ask_order_id = self.current_ask_order_id
                
                self.current_bid_order_id = bid_order.order_id
                self.current_ask_order_id = ask_order.order_id

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            if self.previous_bid_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_bid_order_id )
                to_cancel.append(self.previous_bid_order_id)
            if self.previous_ask_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_ask_order_id )
                to_cancel.append(self.previous_ask_order_id)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders



class LimitMarketStrategy:
    """
        This strategy places limit or market orders every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
            self,
            line_coefficients: Tuple[float, float],
            parabola_coefficients: Tuple[float, float, float],
            trade_size: Optional[float] = 0.001,
            price_tick: Optional[float] = 0.1,
            delay: Optional[int] = 1e8,
            hold_time: Optional[int] = 1e10
    ) -> None:
        """
            Args:
                line_coefficients: line coefficients [k, b] y = kx + b
                parabola_coefficients: parabola coefficients [a, b, c] y = ax^2 + bx + c
                trade_size: volume of each trade
                price_tick: a value by which we increase a bid (reduce an ask) limit order
                delay: delay between orders in nanoseconds
                hold_time: holding time in nanoseconds
        """

        self.trade_size = trade_size
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').delta)
        self.hold_time = hold_time

        # market data list
        self.md_list = []
        # executed trades list
        self.trades_list = []
        # all updates list
        self.updates_list = []

        self.current_time = None
        self.coin_position = 0
        self.coin_position_list = []
        self.prev_midprice = None
        self.current_midprice = None
        self.current_spread = 1
        self.price_tick = price_tick
        self.prev_spread = None
        
        self.line_k = line_coefficients[0]
        self.line_b = line_coefficients[1]
        self.parabola_a, self.parabola_b, self.parabola_c = parabola_coefficients

        self.actions_history = []
        self.actions_history_dict = {}
        
        self.bid_amount = 0
        self.ask_amount = 0
        self.market_making = {}

    def get_normalized_data(self) -> Tuple[float, float]:
        # implement normalization
        return self.coin_position, self.current_spread

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                actions_history: list of tuples(time, coin_pos, spread, action)
        """

        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        while True:
            # get update from simulator
            self.current_time, updates = sim.tick()
            if updates is None:
                break
            # save updates
            self.updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    self.md_list.append(update)
                    self.prev_spread = self.current_spread
                    self.current_midprice = best_ask/2 + best_bid/2
                    self.current_spread = best_ask - best_bid

                elif isinstance(update, OwnTrade):
                    self.trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)

                    if update.side == 'BID':
                        self.coin_position += update.size
                    else:
                        self.coin_position -= update.size
                else:
                    assert False, 'invalid type of update!'

            if self.current_time - prev_time >= self.delay:
                # place order
                inventory, spread = self.get_normalized_data()

                if (self.parabola_a * inventory ** 2 + self.parabola_b * inventory + self.parabola_c) > spread:
                    bid_market_order = sim.place_order(self.current_time, self.trade_size, 'BID', best_ask)
                    ongoing_orders[bid_market_order.order_id] = bid_market_order
                    action = 'market buy'
                elif (self.parabola_a * inventory ** 2 + self.parabola_b * (-inventory) + self.parabola_c) > spread:
                    ask_market_order = sim.place_order(self.current_time, self.trade_size, 'ASK', best_bid)
                    ongoing_orders[ask_market_order.order_id] = ask_market_order
                    action = 'market sell'
                else:
                    ## make self.line_k = 1 + self.coin_position
                    above_line1 = ((self.line_k) *   inventory + self.line_b) < spread
                    above_line2 = ((self.line_k) * (-inventory) + self.line_b) < spread

                    bid_price = best_bid + self.price_tick * above_line1
                    ask_price = best_ask - self.price_tick * above_line2

                    bid_limit_order = sim.place_order(self.current_time, self.trade_size, 'BID', bid_price)
                    ask_limit_order = sim.place_order(self.current_time, self.trade_size, 'ASK', ask_price)
                    ongoing_orders[bid_limit_order.order_id] = bid_limit_order
                    ongoing_orders[ask_limit_order.order_id] = ask_limit_order
                    action = 'limit order'

                prev_time = self.current_time
                self.coin_position_list.append(self.coin_position)
                self.actions_history.append((self.current_time, self.coin_position,
                                             self.current_spread, action))
                self.actions_history_dict[self.current_time] = action
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < self.current_time - self.hold_time:
                    sim.cancel_order(self.current_time, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return self.trades_list, self.md_list, self.updates_list, self.actions_history

class LimitMarketStrategyEnhanced:
    """
        This strategy places limit or market orders every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
            self,
            line_coefficients: Tuple[float, float],
            parabola_coefficients: Tuple[float, float, float],
            coin_position_multiplier: float,
            trade_size: Optional[float] = 0.001,
            price_tick: Optional[float] = 0.1,
            delay: Optional[int] = 1e8,
            hold_time: Optional[int] = 1e10
    ) -> None:
        """
            Args:
                line_coefficients: line coefficients [k, b] y = kx + b
                parabola_coefficients: parabola coefficients [a, b, c] y = ax^2 + bx + c
                trade_size: volume of each trade
                price_tick: a value by which we increase a bid (reduce an ask) limit order
                delay: delay between orders in nanoseconds
                hold_time: holding time in nanoseconds
        """

        self.trade_size = trade_size
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').delta)
        self.hold_time = hold_time

        # market data list
        self.md_list = []
        # executed trades list
        self.trades_list = []
        # all updates list
        self.updates_list = []

        self.current_time = None
        self.coin_position = 0
        self.coin_position_list = []
        self.coin_position_mult = coin_position_multiplier
        self.prev_midprice = None
        self.current_midprice = None
        self.current_spread = 1
        self.price_tick = price_tick
        self.prev_spread = None
        
        self.line_k = line_coefficients[0]
        self.line_b = line_coefficients[1]
        self.parabola_a, self.parabola_b, self.parabola_c = parabola_coefficients

        self.actions_history = []
        self.actions_history_dict = {}

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                actions_history: list of tuples(time, coin_pos, spread, action)
        """

        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        while True:
            # get update from simulator
            self.current_time, updates = sim.tick()
            if updates is None:
                break
            # save updates
            self.updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    self.md_list.append(update)
                    self.prev_spread = self.current_spread
                    self.current_midprice = best_ask/2 + best_bid/2
                    self.current_spread = best_ask - best_bid

                elif isinstance(update, OwnTrade):
                    self.trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)

                    if update.side == 'BID':
                        self.coin_position += update.size
                    else:
                        self.coin_position -= update.size
                else:
                    assert False, 'invalid type of update!'

            if self.current_time - prev_time >= self.delay:
                # place order
                inventory, spread = self.get_normalized_data()

                if (self.parabola_a * inventory ** 2 + self.parabola_b * inventory + self.parabola_c) > spread:
                    bid_market_order = sim.place_order(self.current_time, self.trade_size, 'BID', best_ask)
                    ongoing_orders[bid_market_order.order_id] = bid_market_order
                    action = 'market buy'
                elif (self.parabola_a * inventory ** 2 + self.parabola_b * (-inventory) + self.parabola_c) > spread:
                    ask_market_order = sim.place_order(self.current_time, self.trade_size, 'ASK', best_bid)
                    ongoing_orders[ask_market_order.order_id] = ask_market_order
                    action = 'market sell'
                else:
                    ## make self.line_k = 1 + self.coin_position
                    above_line1 = ((self.line_k + self.coin_position * self.coin_position_mult) *   inventory + self.line_b + self.prev_spread) < spread
                    above_line2 = ((self.line_k + self.coin_position * self.coin_position_mult) * (-inventory) + self.line_b + self.prev_spread) < spread

                    bid_price = best_bid + self.price_tick * above_line1
                    ask_price = best_ask - self.price_tick * above_line2

                    bid_limit_order = sim.place_order(self.current_time, self.trade_size, 'BID', bid_price)
                    ask_limit_order = sim.place_order(self.current_time, self.trade_size, 'ASK', ask_price)
                    ongoing_orders[bid_limit_order.order_id] = bid_limit_order
                    ongoing_orders[ask_limit_order.order_id] = ask_limit_order
                    action = 'limit order'

                prev_time = self.current_time
                self.coin_position_list.append(self.coin_position)
                self.actions_history.append((self.current_time, self.coin_position,
                                             self.current_spread, action))
                self.actions_history_dict[self.current_time] = action

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < self.current_time - self.hold_time:
                    sim.cancel_order(self.current_time, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return self.trades_list, self.md_list, self.updates_list, self.actions_history

class StoikovStrategyGeneralizedSingleAsset:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, trade_size: float, position_limit: float, delay: float, hold_time:Optional[float] = None, risk_aversion:Optional[float] = 0.5) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        self.order_size = trade_size
        self.last_mid_prices = []
        self.gamma = risk_aversion
        self.Q = position_limit
        self.asset_position = 0
        self.current_bid_order_id = None
        self.current_ask_order_id = None
        self.previous_bid_order_id = None
        self.previous_ask_order_id = None
        self.trades_dict = {'place_ts' :[], 'exchange_ts': [], 'receive_ts': [], 'trade_id': [],'order_id': [],'side': [], 'size': [], 'price': [],'execute':[], 'mid_price':[]}  
    
    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    mid_price = (best_bid + best_ask)/2
                    if len(self.last_mid_prices) < 500:
                        self.last_mid_prices.append(mid_price)
                    else:
                        self.last_mid_prices.append(mid_price)
                        self.last_mid_prices.pop(0)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    self.trades_dict['place_ts'].append(update.place_ts)
                    self.trades_dict['exchange_ts'].append(update.exchange_ts)
                    self.trades_dict['receive_ts'].append(update.receive_ts)
                    self.trades_dict['trade_id'].append(update.trade_id)
                    self.trades_dict['order_id'].append(update.order_id)
                    self.trades_dict['side'].append(update.side)
                    self.trades_dict['size'].append(update.size)
                    self.trades_dict['price'].append(update.price)
                    self.trades_dict['execute'].append(update.execute)
                    self.trades_dict['mid_price'].append(mid_price)
                    if update.side == "ASK":
                        self.asset_position -= update.size
                    elif update.side == "BID":
                        self.asset_position += update.size
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                #place order
                '''
                reservation_price = s - q * gamma * (sigma**2) * (T - t)
                delta_bid and delta_ask are equivalently distant from the reservation_orice
                delta_bid + delta_ask = gamma * (sigma**2) * (T-t) + 2/gamma * ln(1 + gamma/k)
                k = K*alpha
                
                s      : current mid_price
                q      : current position in asset
                sigma  : parameter in the Geometric Brownian Motion equation [dS_t = sigma dw_t]
                gamma  : risk-aversion parameter of the optimizing agents (across the economy)
                xi     : often referred to as the gamma parameter, but has slightly different meaning - the magnitude towards exponential utility function over linear utility function
                delta  : or equvalently the size of limit orders
                K      : higher K means that market order volumes have higher impact on best price changes
                alpha  : higher alpha means higher probability fo large market orders
                A      : scaling parameter in the density function of market order size 
                Q      : limit of position, the market maker stops providing the limit orders that could make the asset_position violate the limit
                
                '''
                xi = self.gamma
                if len(self.last_mid_prices)==500:
                    sigma = np.std(self.last_mid_prices)## per update --> need to scale it to the "per second" terminology
                else:
                    sigma = 1
                sigma = sigma*np.sqrt(1/0.032)
                delta_t = 0.032 ## there is approximately 0.032 seconds in between the orderbook uprates (nanoseconds / 1e9 = seconds)
                k = 1.5
                A = 1
                q = self.asset_position
                delta_ = self.order_size
                ## mid_price was defined previously
                delta_ask = 1/xi/delta_ * np.log(1 + xi*delta_/k) - (2*q - delta_)/2*np.sqrt( self.gamma*(sigma**2)/2/A/delta_/k * (1+xi*delta_/k)**(1+k/xi/delta_) )
                delta_bid = 1/xi/delta_ * np.log(1 + xi*delta_/k) + (2*q + delta_)/2*np.sqrt( self.gamma*(sigma**2)/2/A/delta_/k * (1+xi*delta_/k)**(1+k/xi/delta_) )
                
                bid_price = np.round(mid_price - delta_bid, 1)
                ask_price = np.round(mid_price + delta_ask, 1)
                if (self.asset_position < self.Q):
                    bid_order = sim.place_order( receive_ts, self.order_size, 'BID', bid_price)
                    ongoing_orders[bid_order.order_id] = bid_order
                    self.previous_bid_order_id = self.current_bid_order_id
                    self.current_bid_order_id = bid_order.order_id
                    all_orders.append(bid_order)
                    
                if (self.asset_position > -self.Q):
                    ask_order = sim.place_order( receive_ts, self.order_size, 'ASK', ask_price)
                    ongoing_orders[ask_order.order_id] = ask_order
                    self.previous_ask_order_id = self.current_ask_order_id
                    self.current_ask_order_id = ask_order.order_id
                    all_orders.append(ask_order)
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            if self.previous_bid_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_bid_order_id )
                to_cancel.append(self.previous_bid_order_id)
            if self.previous_ask_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_ask_order_id )
                to_cancel.append(self.previous_ask_order_id)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders
    
    