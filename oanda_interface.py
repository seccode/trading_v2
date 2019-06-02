
from datetime import datetime
import json
import oandapyV20
import numpy as np
import oandapyV20.endpoints.forexlabs as labs
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails, TrailingStopLossDetails
from oandapyV20.contrib.requests import TradeCloseRequest,MarketOrderRequest

class oanda_interface():
    '''Class to handle all communication with oanda server'''

    def __init__(self,account_id,access_token,environment='live'):
        '''Initializes client connection with oanda

        Parameters
        ----------
        account_id : str
            Account id for client set up
        access_token : str
            The API access token
        enviornment : str,optional
            Whether the enviornment is 'practice' or 'live' (default is live)
        '''
        self.account_id = account_id
        self.access_token = access_token
        self.API = oandapyV20.API(access_token=self.access_token,environment=environment)

    def get_account_information(self):
        '''Returns all information about account given in init'''
        # 1. Make request format
        request = oandapyV20.endpoints.accounts.AccountDetails(self.account_id)
        # 2. Send request
        self.API.request(request)
        # 4 . Return result
        return request.response

    def get_instrument_info(self,instrument,granularity,number_of_candles,start_time=None,end_time=None):
        '''gets number_of_candles from start_time (if specified) for instrument at granularity
        returns dictionary of lists dictionary keys are 'time','bid_open','bid_high','bid_low','bid_close','ask_open','ask_high','ask_low','ask_close'
        volume list for each are of length number of candles, and represent the values for each of the keys, at that time'''
        # 1. Make request format
        if start_time == None and end_time == None:
            params = {'granularity':granularity,'count':number_of_candles,'price':'BA'}
        elif end_time == None:
            params = {'granularity':granularity,'count':number_of_candles,'price':'BA','from':start_time}
        elif start_time == None:
            params = {'granularity':granularity,'count':number_of_candles,'price':'BA','to':end_time}

        request = instruments.InstrumentsCandles(instrument=instrument,params=params)
        # 2. Send request
        self.API.request(request)
        # 3. Change results into dictionary output
        return_dict = {'time':[],'bid_open':[],'bid_high':[],'bid_low':[],'bid_close':[],'ask_open':[],'ask_high':[],'ask_low':[],'ask_close':[],'volume':[]}
        for candle in request.response['candles']:
            return_dict['time'].append(candle['time'])
            return_dict['volume'].append(float(candle['volume']))
            return_dict['bid_open'].append(float(candle['bid']['o']))
            return_dict['bid_high'].append(float(candle['bid']['h']))
            return_dict['bid_close'].append(float(candle['bid']['c']))
            return_dict['bid_low'].append(float(candle['bid']['l']))
            return_dict['ask_open'].append(float(candle['ask']['o']))
            return_dict['ask_high'].append(float(candle['ask']['h']))
            return_dict['ask_close'].append(float(candle['ask']['c']))
            return_dict['ask_low'].append(float(candle['ask']['l']))
        return return_dict # Last entry is most recent

    def get_order_book(self,instrument,time):
        '''gets the order book at time for instruments_to_get, returns nx4 numpy array, columns are price, long count percent, short count percent'''
        # 1. Make request format
        params = {'time':time}
        request = instruments.InstrumentsOrderBook(instrument=instrument,params=params)
        # 2. Send request
        self.API.request(request)
        # 3. Return in desired format
        return_mat = []
        for bucket in request.response['orderBook']['buckets']:
            return_mat.append([bucket['price'],bucket['longCountPercent'],bucket['shortCountPercent']])
        return np.array(return_mat).astype(float)

    def get_position_book(self,instrument,time):
        '''gets the position book at time for instruments_to_get, returns nx4 numpy array, columns are price, long count percent, short count percent'''
        # 1. Make request format
        params = {'time':time}
        request = instruments.InstrumentsPositionBook(instrument=instrument,params=params)
        # 2. Send request
        self.API.request(request)
        # 3. Return in desired format
        return_mat = []
        for bucket in request.response['positionBook']['buckets']:
            return_mat.append([bucket['price'],bucket['longCountPercent'],bucket['shortCountPercent']])
        print(np.array(return_mat).astype(float).shape)
        return np.array(return_mat).astype(float)

    def open_position(self,instrument,size,type_of_trade,take_profit=None,stop_loss=None,trail_stop=None):
        '''Opens a market order position

        Parameters
        ----------
        instrument : str
            What instrument to open up position for
        size : int
            size of position to open
        type_of_trade : str
            Either 'long' or 'short' -- type of position to take
        take_profit : float, optional
            If set by call, will try and place market order with the take profit
        stop_loss : float, optional
            If set by call, will try and place market order with the stop_loss
        trail_stop : float, optional
            If set by call, will try and place market order with the trail stop loss

        Returns
        ----------
        trade_id : int
            Id of succesful trade
        '''
        # 1. Set up position size based on trade type
        if type_of_trade == 'short':
            size = -size

        # 2. Set market order
        if not take_profit and not stop_loss and not trail_stop: # none set
            mktOrder = MarketOrderRequest(
                instrument=instrument,
                units=size)
        if take_profit and (not stop_loss and not trail_stop):   # just take
            mktOrder = MarketOrderRequest(
                instrument=instrument,
                units=size,
                takeProfitOnFill=TakeProfitDetails(price=take_profit).data)
        if stop_loss and (not take_profit and not trail_stop):   # just stop
            mktOrder = MarketOrderRequest(
                instrument=instrument,
                units=size,
                stopLossOnFill=StopLossDetails(price=stop_loss).data)
        if trail_stop and (not stop_loss and not take_profit):   # just trail
            mktOrder = MarketOrderRequest(
                instrument=instrument,
                units=size,
                trailingStopLossOnFill=TrailingStopLossDetails(distance=trail_stop).data)

        if take_profit and stop_loss and not trail_stop:         # take and stop
            mktOrder = MarketOrderRequest(
                instrument=instrument,
                units=size,
                takeProfitOnFill=TakeProfitDetails(price=take_profit).data,
                stopLossOnFill=StopLossDetails(price=stop_loss).data)

        if take_profit and trail_stop and not stop_loss:         # take and trail
            mktOrder = MarketOrderRequest(
                instrument=instrument,
                units=size,
                takeProfitOnFill=TakeProfitDetails(price=take_profit).data,
                trailingStopLossOnFill=TrailingStopLossDetails(distance=trail_stop).data)

        if stop_loss and trail_stop and not take_profit:         # stop and trail
            mktOrder = MarketOrderRequest(
                instrument=instrument,
                units=size,
                stopLossOnFill=StopLossDetails(price=stop_loss).data,
                trailingStopLossOnFill=TrailingStopLossDetails(distance=trail_stop).data)

        if take_profit and trail_stop and stop_loss:             # take, trail, and stop
            mktOrder = MarketOrderRequest(
                instrument=instrument,
                units=size,
                takeProfitOnFill=TakeProfitDetails(price=take_profit).data,
                stopLossOnFill=StopLossDetails(price=stop_loss).data,
                trailingStopLossOnFill=TrailingStopLossDetails(distance=trail_stop).data)


        # 3. Make request
        r = orders.OrderCreate(self.account_id, data=mktOrder.data)
        rv = self.API.request(r)
        # print(rv)

        # print(rv['orderFillTransaction']['tradeOpened']['tradeID'])
        try:
            return rv['orderFillTransaction']['tradeOpened']['tradeID']
        except:
            pass
        try:
            return rv['orderFillTransaction']['tradeReduced']['tradeID']
        except:
            pass
        try:
            return rv['orderFillTransaction']['tradeClosed']['tradeID']
        except:
            pass
        # try:
        #     # create the OrderCreate request
        #     print('here')
        #     rv = self.API.request(r)
        # except oandapyV20.exceptions.V20Error as err:
        #     print(r.status_code, err)
        # else:
        #     print(rv)
        #     return rv['orderFillTransaction']['tradeOpened']['tradeID']

    def close_position(self,trade_id):
        '''Closes trade with trade_id at market price '''
        order = TradeCloseRequest()
        r = trades.TradeClose(self.account_id,tradeID=trade_id,data=order.data)
        rv = self.API.request(r)

    def check_order_active(self,trade_id):
        '''Given the trade_id checks to see if that trade is still active'''
        r = orders.OrderList(self.account_id)
        rv = self.API.request(r)
        active_ids = []
        for order in rv['orders']:
            active_ids.append(order['tradeID'])
        return True if trade_id in active_ids else False

    def get_trades(self):
        '''Gets all trades from account'''
        r = trades.TradesList(accountID=self.account_id)
        self.API.request(r)
        print(r.response)

    def get_calendar_info(self,instrument,length):
        '''Gets historical calendar info about instrument'''
        r = labs.Calendar(params={"instrument":instrument,"period":length})
        self.API.request(r)
        return r.response
        









if __name__ == '__main__':
    tradeInterface = oanda_interface('101-001-8734770-001','badcf7760c1558e67da9dcc7144be117-e4dc07376ebb83f374fb3a7cb82725f6',environment='practice')
    #tradeInterface = oanda_interface('001-001-2154685-001','83bfe1b504bd65b07513a811b630993f-cbf32b5d40d2bc7827f11308d2a5b55c')
    #tradeInterface.get_trades()
    #m = tradeInterface.get_account_information()
    #print(m['account']['balance'])
    #m = tradeInterface.get_instrument_info('EUR_USD','S5',1)#,start_time = '2005-01-01T08:10:00.000000000Z')
    #print(m)
    #tradeInterface.get_position_book('EUR_USD',"2018-01-10T14:40:00Z")
    print(tradeInterface.open_position('EUR_USD',1000,'long'))#,take_profit=1.132,stop_loss=1.130))#,trail_stop=.0005)#,take_profit=None,stop_loss=None,trail_stop=None)
    #tradeInterface.close_position(1116280)
    #print(tradeInterface.check_order_active('1116289'))

'''
necessary functions:

open_position(insrument, size, long/short):
    ensures position size is valid
    runs until position succesfully opened
    returns trade id

close_position(trade_id):
    closes trade with id trade_id


get_account_info():
    returns all account info as dictionary

get_instrument_info(instrument,granularity,number_of_candles,start_time):
    gets number_of_candles from start_time (if specified) for instrument at granularity
    returns dictionary of lists
    dictionary keys are 'time','bid_open','bid_high','bid_low','bid_close','ask_open','ask_high','ask_low','ask_close'
    list for each are of length number of candles, and represent the values for each of the keys, at that time

get_order_book(instrument,bucket_width,time)
    gets the order book at time, with bucket width of bucket_width for instruments_to_get

get_position_book(instrument,bucket_width,time)
    gets the position book at time, with bucket width of bucket_width for instruments_to_get



'''


#
