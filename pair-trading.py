# coding: utf-8
# ##################################################################
# Pair Trading adapted to backtrader
# with PD.OLS and info for StatsModel.API
# author: Remi Roche
##################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
from ast import arg
import datetime

# The above could be sent to an independent module
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
import numpy as np

# my import
import jqdatasdk
from jqdatasdk import *


class PairTradingStrategy(bt.Strategy):
    params = dict(
        period=10,
        stake=10,
        qty1=0,
        qty2=0,
        printout=True,
        upper=2.1,
        lower=-2.1,
        up_medium=0.5,
        low_medium=-0.5,
        status=0,
        portfolio_value=10000,
    )

    def log(self, txt, dt=None):
        if self.p.printout:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        """
        提示订单信息
        """
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return  # Await further notifications

        if order.status == order.Completed:
            if order.isbuy():
                buytxt = 'BUY COMPLETE, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price,
                     order.executed.value,
                     order.executed.comm)
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.log(buytxt, order.executed.dt)
            else:
                selltxt = 'SELL COMPLETE, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price,
                          order.executed.value,
                          order.executed.comm)
                self.bar_executed = len(self)
                self.sellprice = order.executed.price
                self.sellcomm = order.executed.comm
                self.log(selltxt, order.executed.dt)

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            self.log('%s ,' % order.Status[order.status])
            pass  # Simply log

        # Allow new orders
        self.orderid = None

    def __init__(self):
        # To control operation entries
        self.orderid = None
        self.qty1 = self.p.qty1
        self.qty2 = self.p.qty2
        self.upper_limit = self.p.upper
        self.lower_limit = self.p.lower
        self.up_medium = self.p.up_medium
        self.low_medium = self.p.low_medium
        self.status = self.p.status
        self.portfolio_value = self.p.portfolio_value

        # Signals performed with PD.OLS : 
        self.transform = btind.OLS_TransformationN(self.data0, self.data1,
                                                   period=self.p.period)
        self.zscore = self.transform.zscore

        # cointN返回两个序列协整的p值
        # self.transform = btind.CointN(self.data0, self.data1,
        #                                             period=self.p.period)
        # tmp=self.transform.lines.score
        # print(tmp,"\n\n")
        # self.zscore = self.transform.zscore        

        # Checking signals built with StatsModel.API :
        # self.ols_transfo = btind.OLS_Transformation(self.data0, self.data1,
        #                                             period=self.p.period,
        #                                             plot=True)

    def next(self):

        if self.orderid:
            return  # if an order is active, no new orders are allowed

        if self.p.printout:
            print('Self  len:', len(self))
            print('Data0 len:', len(self.data0))
            print('Data1 len:', len(self.data1))
            print('Data0 len == Data1 len:',
                  len(self.data0) == len(self.data1))

            print('Data0 dt:', self.data0.datetime.datetime())
            print('Data1 dt:', self.data1.datetime.datetime())

        print('status is', self.status)
        print('zscore is', self.zscore[0])

        # Step 2: Check conditions for SHORT & place the order
        # Checking the condition for SHORT
        if (self.zscore[0] > self.upper_limit) and (self.status != 1):

            # Calculating the number of shares for each stock
            value = 0.5 * self.portfolio_value  # Divide the cash equally
            x = int(value / (self.data0.close))  # Find the number of shares for Stock1
            y = int(value / (self.data1.close))  # Find the number of shares for Stock2
            print('x + self.qty1 is', x + self.qty1)
            print('y + self.qty2 is', y + self.qty2)

            # Placing the order
            self.log('SELL CREATE %s, price = %.2f, qty = %d' % ("PEP", self.data0.close[0], x + self.qty1))
            self.sell(data=self.data0, size=(x + self.qty1))  # Place an order for buying y + qty2 shares
            self.log('BUY CREATE %s, price = %.2f, qty = %d' % ("KO", self.data1.close[0], y + self.qty2))
            self.buy(data=self.data1, size=(y + self.qty2))  # Place an order for selling x + qty1 shares

            # Updating the counters with new value
            self.qty1 = x  # The new open position quantity for Stock1 is x shares
            self.qty2 = y  # The new open position quantity for Stock2 is y shares

            self.status = 1  # The current status is "short the spread"

            # Step 3: Check conditions for LONG & place the order
            # Checking the condition for LONG
        elif (self.zscore[0] < self.lower_limit) and (self.status != 2):

            # Calculating the number of shares for each stock
            value = 0.5 * self.portfolio_value  # Divide the cash equally
            x = int(value / (self.data0.close))  # Find the number of shares for Stock1
            y = int(value / (self.data1.close))  # Find the number of shares for Stock2
            print('x + self.qty1 is', x + self.qty1)
            print('y + self.qty2 is', y + self.qty2)

            # Place the order
            self.log('BUY CREATE %s, price = %.2f, qty = %d' % ("PEP", self.data0.close[0], x + self.qty1))
            self.buy(data=self.data0, size=(x + self.qty1))  # Place an order for buying x + qty1 shares
            self.log('SELL CREATE %s, price = %.2f, qty = %d' % ("KO", self.data1.close[0], y + self.qty2))
            self.sell(data=self.data1, size=(y + self.qty2))  # Place an order for selling y + qty2 shares

            # Updating the counters with new value
            self.qty1 = x  # The new open position quantity for Stock1 is x shares
            self.qty2 = y  # The new open position quantity for Stock2 is y shares
            self.status = 2  # The current status is "long the spread"


            # Step 4: Check conditions for No Trade
            # If the z-score is within the two bounds, close all
        """
        elif (self.zscore[0] < self.up_medium and self.zscore[0] > self.low_medium):
            self.log('CLOSE LONG %s, price = %.2f' % ("PEP", self.data0.close[0]))
            self.close(self.data0)
            self.log('CLOSE LONG %s, price = %.2f' % ("KO", self.data1.close[0]))
            self.close(self.data1)
        """

    def stop(self):
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')


def runstrategy():
    # todo 参数用起来
    args = parse_args()

    # Create a cerebro
    cerebro = bt.Cerebro()

    # Get the dates from the args
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')

    # Create the 1st data
    # data0 = btfeeds.YahooFinanceCSVData(
    #     dataname=args.data0,
    #     fromdate=fromdate,
    #     todate=todate)、
    stock0=args.data0
    testdata=jqdatasdk.get_price('600519.XSHG',start_date='2021-07-01',end_date='2021-12-31',frequency='daily',fields=["open"	,"high"	,"low"	,"close"	,"volume"])
    testdata=testdata.reset_index().rename(columns={"index":"Date"	,"open":"Open"	,"close":"Close"	,"high":"High"	,"low":"Low"	,"volume":"Volume"}).set_index("Date")
    data0=bt.feeds.PandasData(dataname=testdata)

    # Add the 1st data to cerebro
    cerebro.adddata(data0)

    # Create the 2nd data
    # data1 = btfeeds.YahooFinanceCSVData(
    #     dataname=args.data1,
    #     fromdate=fromdate,
    #     todate=todate)
    stock1=args.data1
    testdata=jqdatasdk.get_price("601318.XSHG",start_date='2021-07-01',end_date='2021-12-31',frequency='daily',fields=["open"	,"high"	,"low"	,"close"	,"volume"])
    testdata=testdata.reset_index().rename(columns={"index":"Date"	,"open":"Open"	,"close":"Close"	,"high":"High"	,"low":"Low"	,"volume":"Volume"}).set_index("Date")
    data1=bt.feeds.PandasData(dataname=testdata)

    # Add the 2nd data to cerebro
    cerebro.adddata(data1)

    # Add the strategy
    cerebro.addstrategy(PairTradingStrategy,
                        period=args.period,
                        stake=args.stake)

    # Add the commission - only stocks like a for each operation
    cerebro.broker.setcash(args.cash)

    # Add the commission - only stocks like a for each operation
    cerebro.broker.setcommission(commission=args.commperc)

    # And run it
    cerebro.run(runonce=not args.runnext,
                preload=not args.nopreload,
                oldsync=args.oldsync)

    # Plot if requested
    if args.plot:
        cerebro.plot(numfigs=args.numfigs, volume=False, zdown=False)


def parse_args():
    parser = argparse.ArgumentParser(description='MultiData Strategy')

    parser.add_argument('--data0', '-d0',
                        default='../../datas/daily-PEP.csv',
                        help='1st data into the system')

    parser.add_argument('--data1', '-d1',
                        default='../../datas/daily-KO.csv',
                        help='2nd data into the system')

    parser.add_argument('--fromdate', '-f',
                        default='1997-01-01',
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--todate', '-t',
                        default='1998-06-01',
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--period', default=10, type=int,
                        help='Period to apply to the Simple Moving Average')

    parser.add_argument('--cash', default=100000, type=int,
                        help='Starting Cash')

    parser.add_argument('--runnext', action='store_true',
                        help='Use next by next instead of runonce')

    parser.add_argument('--nopreload', action='store_true',
                        help='Do not preload the data')

    parser.add_argument('--oldsync', action='store_true',
                        help='Use old data synchronization method')

    parser.add_argument('--commperc', default=0.005, type=float,
                        help='Percentage commission (0.005 is 0.5%%')

    parser.add_argument('--stake', default=10, type=int,
                        help='每次交易的股数Stake to apply in each operation')

    parser.add_argument('--plot', '-p', default=True, action='store_true',
                        help='Plot the read data')

    parser.add_argument('--numfigs', '-n', default=1,
                        help='Plot using numfigs figures')

    return parser.parse_args()


"""另一个"""
class KalmanPair(bt.Strategy):
    params = (("printlog", False), ("quantity", 1000))

    def log(self, txt, dt=None, doprint=False):
        """Logging function for strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")

    def __init__(self):
        self.delta = 0.0001
        self.Vw = self.delta / (1 - self.delta) * np.eye(2)
        self.Ve = 0.001

        self.beta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = np.zeros((2, 2))

        self.position_type = None  # long or short
        self.quantity = self.params.quantity

    def next(self):

        x = np.asarray([self.data0[0], 1.0]).reshape((1, 2))
        y = self.data1[0]

        self.R = self.P + self.Vw  # state covariance prediction
        yhat = x.dot(self.beta)  # measurement prediction

        Q = x.dot(self.R).dot(x.T) + self.Ve  # measurement variance

        e = y - yhat  # measurement prediction error

        K = self.R.dot(x.T) / Q  # Kalman gain

        self.beta += K.flatten() * e  # State update
        self.P = self.R - K * x.dot(self.R)

        sqrt_Q = np.sqrt(Q)

        if self.position:
            if self.position_type == "long" and e > -sqrt_Q:
                self.close(self.data0)
                self.close(self.data1)
                self.position_type = None
            if self.position_type == "short" and e < sqrt_Q:
                self.close(self.data0)
                self.close(self.data1)
                self.position_type = None

        else:
            if e < -sqrt_Q:
                self.sell(data=self.data0, size=(self.quantity * self.beta[0]))
                self.buy(data=self.data1, size=self.quantity)

                self.position_type = "long"
            if e > sqrt_Q:
                self.buy(data=self.data0, size=(self.quantity * self.beta[0]))
                self.sell(data=self.data1, size=self.quantity)
                self.position_type = "short"

        self.log(f"beta: {self.beta[0]}, alpha: {self.beta[1]}")


def run():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(KalmanPair)

    startdate = datetime.datetime(2007, 1, 1)
    enddate = datetime.datetime(2017, 1, 1)

    testdata=jqdatasdk.get_price('600519.XSHG',start_date='2021-07-01',end_date='2021-12-31',frequency='daily',fields=["open"	,"high"	,"low"	,"close"	,"volume"])
    testdata=testdata.reset_index().rename(columns={"index":"Date"	,"open":"Open"	,"close":"Close"	,"high":"High"	,"low":"Low"	,"volume":"Volume"}).set_index("Date")
    data0=bt.feeds.PandasData(dataname=testdata)

    # Add the 1st data to cerebro
    cerebro.adddata(data0)

    # Create the 2nd data
    testdata=jqdatasdk.get_price("601318.XSHG",start_date='2021-07-01',end_date='2021-12-31',frequency='daily',fields=["open"	,"high"	,"low"	,"close"	,"volume"])
    testdata=testdata.reset_index().rename(columns={"index":"Date"	,"open":"Open"	,"close":"Close"	,"high":"High"	,"low":"Low"	,"volume":"Volume"}).set_index("Date")
    data1=bt.feeds.PandasData(dataname=testdata)

    # Add the 2nd data to cerebro
    cerebro.adddata(data1)

    # ewa = bt.feeds.YahooFinanceData(dataname="EWA", fromdate=startdate, todate=enddate)
    # ewc = bt.feeds.YahooFinanceData(dataname="EWC", fromdate=startdate, todate=enddate)

    # cerebro.adddata(ewa)
    # cerebro.adddata(ewc)

    # #cerebro.broker.setcommission(commission=0.0001)
    cerebro.broker.setcash(100_000.0)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()


if __name__ == '__main__':
    auth('15857500957','Qazwsxedcrfv0957')
    run()
