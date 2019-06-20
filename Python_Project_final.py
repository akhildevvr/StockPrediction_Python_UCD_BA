# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 01:23:56 2018

@author: Jose Chiramel
"""

import statsmodels.api as sm
import sys
from scipy.stats import variation
#from scipy.stats import linregress
import pandas as pd
import numpy as np
data = pd.read_csv("http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchan0ge=nasdaq&render=download")
import pandas_datareader as web
import matplotlib.pyplot as plt

from pandas import Series
from scipy.stats import variation
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader as web
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import  timedelta
from sklearn.metrics import mean_squared_error
import math

#from datetime import datetime,timedelta
#import quandl as quandl
symbol1=data['Symbol'].tolist()
symbol= np.array(data['Symbol'])
bool = True

def prepare_data():
    start_date = input("PLEASE ENTER THE START DATE FOR MODELLING(i.e.(%y,%m,%d) 2018/01/01) : ")
    year, month, day = map(int, start_date.split('/'))
    start_date = datetime.datetime(year, month, day)
    end_date = input("PLEASE ENTER THE END DATE FOR MODELLING(i.e.(%y,%m,%d) 2018/01/01) : ")
    year, month, day = map(int, end_date.split('/'))
    end_date = datetime.datetime(year, month, day)
    data2 = web.DataReader(code, "yahoo", start = start_date ,end= end_date)
    data2
    data2 = pd.DataFrame(data2)
    print(data2)
# create train test partition
    data2 = data2.reset_index()
    close = data2['Close'].tolist()
    dates = data2.index.tolist()
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(close, (len(close), 1))
    intel_regress(dates,prices,end_date)

#Convert to 1d Vector


def intel_regress(dates,prices,end_date):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(dates, prices)
    dummy_regressor=regressor
# Predicting the Test set results
    y_pred = regressor.predict(dates)
    print('Coefficients: ', regressor.coef_[0][0], '\n')
# The mean square error
    print("Residual sum of squares: %.2f"
      % np.mean((regressor.predict(dates) - prices) ** 2),'\n')
# Explained variance score: 1 is perfect prediction
    print('The coefficient of determination : %.2f' %regressor.score(dates,prices ),'\n')
    mse = mean_squared_error(y_pred, prices)
    rmse =math.sqrt(mse)
    print('Root Mean square value : %.2f' % rmse,'\n' )
    plt.scatter(dates, prices, color='red') #plotting the initial datapoints
    plt.plot(dates, regressor.predict(dates), color='blue', linewidth=3) #plotting the line made by linear regression
    plt.title('Linear Regression : Time vs. Price')
    plt.legend()
    plt.show()
    end_date=end_date
    intel_predict(dummy_regressor,end_date)
    


def intel_predict(dummy_regressor,end_date):
    date1 = input('Enter a date in YYYY/MM/DD format for prediction :  ')
    year, month, day = map(int, date1.split('/'))
    date1 = datetime.datetime(year, month, day)

    if date1 >= datetime.datetime.now():
        date2 = date1 - end_date
        no_of_days= date2.days
        predicted_price = dummy_regressor.predict(no_of_days)
        print("The Closing Value for the {code} is : {predicted_price} " .format(code=code, predicted_price = predicted_price[0][0]),'\n')
    else:
        date3 = datetime.datetime.now() - date1
        no_of_days1 = date3.days
        predicted_price1 =dummy_regressor.predict(no_of_days1)
        print("The Closing Value for  {code} is : {predicted_price} " .format(code=code, predicted_price = predicted_price1[0][0]),'\n')

    
def start1(code,start,end):
    data = web.DataReader(code,"yahoo",start = start ,end = end)
    if len(data) < 1:
        return False
    else:
        return data
      
    
def descriptive(df):
    return df2.describe()
    return variation(df2)

def descriptive_graph(df):
    plt.figure(figsize=(15,10))                                   
    df1.plot(grid =True)
    plt.title( code + " stock price")
    plt.ylabel("Price($)")
    plt.show()   
    print("="*100)  

def time_seris(df):
#Graphical Data Sets
#Closing price raw time-series
    print("These are " + code + "'s daily closing price within the required timeframe")
    plt.figure(figsize=(15,10))                                   
    df1.plot(grid =True)
    plt.title( code + " stock price")
    plt.ylabel("Price($)")
    plt.show()   
    print("="*100)    

def trend(df):
#Trend line
    print("This is the trend line for " + code)
    data1 = sm.OLS(df1,sm.add_constant(range(len(df1.index)),prepend=True)).fit().fittedvalues
    plt.figure(figsize=(15,10))
    df1.plot(grid =True)
    plt.plot(data1, label="trend line")
    plt.show()
    print("="*100) 

def macd(df):
#MACD
     print(code +"'s" + " Moving average convergence/divergence")
     plt.figure(figsize=(15,10))
     plt.grid(True)
     close_26_ewma = df1.ewm(span=26,min_periods=0, adjust=True, ignore_na=True).mean()
     close_12_ewma = df1.ewm(span=12,min_periods=0, adjust=True, ignore_na=True).mean()
     data_26 = close_26_ewma
     data_12 = close_12_ewma
     data_macd = data_12 - data_26
     plt.plot(data_26,label="EMA_26_days")
     plt.plot(data_12,label="EMA_12_days")
     plt.plot(data_macd,label="MACD")
     plt.legend(loc=2)
     plt.title(code + "'s Moving average convergence/divergence")
     plt.ylabel("Price($)")
     plt.show()                    
     print ("="*100)

def moving_plotting(df):
    choice = input("DO YOU WANT TO ADD MOVING AVERAGE TO YOUR GRAPH? \nEnter y/n:  ")
    if choice =='y':
        x = input("ENTER MOVING AVERAGE WINDOW(i.e 30 = 30 days): ")   #allow users to choose the length of moving average 
        x = int(x)
        df['Moving average'] = df1.rolling(x).mean()                   #rolling 
        plt.figure(figsize=(15,10))
        plt.plot(df2["Moving average"],label='MA '+ str(x) + 'days') 
        df1.plot(grid =True)
        plt.legend(loc=2)
        plt.title(code + "'s " + str(x) + "days moving average")
        plt.ylabel("Price($)")
        plt.show()                    
        print("="*100)
        
        time_seris(df)
        trend(df)
        macd(df)

    
    elif choice =="n":
        time_seris(df)
        trend(df)
        macd(df)
        
    else:
        print("Invalid answer. Please type in y or n")
        moving_plotting(df)
        

def menu1():             #sub menu for yes or not question(quit the program or go back to main menu)                              
        goback_choice = input("DO YOU WANT TO GO BACK TO MAIN MENU? \nENTER y/n:")
        if goback_choice == "y":
            main_menu(df,rerun = False)
        
        elif goback_choice =="n":
            print("End of analysis. See you again in the future.")
            
        else:
            print("Invalid answer. Please type in y or n")
            menu1()
            
        
     
    
def main_menu(df,rerun,rerun1=None):
    if rerun1 != True:
        if rerun == True:
            return False
        else:
            print("What in-depth financial data do you need?")
            print("\n\t1. Descriptive data\n\t2. Graphical Data Sets\n\t3. Predictive analysis\n\t4. Serch for another company(enter \"4\" twice to make sure) \n\t5. Quit")
            choice = input("Please choose option: ")
            if choice == "1":
                print("Below are 's closing price descriptive data")
                print(descriptive(df))
                menu1()
                    
                    
            
            elif choice == "2":
                moving_plotting(df)
                print("Going back to go back to main menu")
                main_menu(df, rerun = False)
                
            elif choice == "3":
                print("predictive stuff, not yet done")
                prepare_data()
                
                
            elif choice == "4":
                return False
        
            
            elif choice == "5":
                print("Goodbye! Hope to you see soon.")
                sys.exit()
                
            
            else:
                print("Wrong Input, Please try again")
                main_menu(df,rerun = False)
    else:
         return False
      

rerun = False
rerun1 = False
while True:
    while True:
        try:    
            code ="global"
            code = input(" enter the company symbol: ")
            if code not in symbol:
                raise ValueError
            else:
                break
        except ValueError:
            print("error")

    while bool:
         try:
             start_date = input("PLEASE ENTER THE START DATE FOR ANALYSIS(i.e.(%y,%m,%d) 2018/01/01) : ")
             end_date = input("PLEASE ENTER THE END DATE FOR ANALYSIS(i.e.(%y,%m,%d) 2018/01/01): ")
             df = start1(code,start_date,end_date) 
             if len(df)>1:
                 break
             elif int(start_date) == int(end_date):
                 raise ValueError
             else:
                 raise ValueError
         except ValueError:
             print("date entered invalid")
                
    df1 = df['Adj Close']                    #Only focus on the adjusted closing column  
    df2 = df                                 #All the data sets will be captured
    df1_close = pd.DataFrame(df1)      #For Moving average
    data_26= pd.DataFrame(df1)         #For MACD
    data_12=pd.DataFrame(df1)          #For MACD
    data_macd=pd.DataFrame(df1)        #For MACD
    
    try:
        main_menu(df,rerun,rerun1)
        if main_menu(df,rerun = False) != False:
            break
        else: 
            rerun1 = True
            raise ValueError
    except ValueError: 
        print("Re-running....")