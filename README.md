# Stock Prediction Using Python

Introduction

Buying and selling of stocks (or bonds and other securities) is a potentially protable business, but can also
lead to big fnancial losses. Business Analytics can provide insights to better inform choices in this industry.
When designing a stock portfolio, some of the decisions to take are:
1. The capital to invest;
2. The risk the investor is willing to take;
3. The time the investor will wait for his/her returns.
The use of analytics methods can help with some of these decisions.

Assignment Details

The task is to write a computer program, which allows users to consult/analyse/model stock times-series.
It will source its data online, but optimise its use of downloaded data, to avoid excessive network traffic.
Users will have the possibility of searching for specific stocks, and query specified time ranges, along
with associated analysis, such as statistical descriptions of prices and/or volume (mean, median, range, etc),
technical indicators, visualisation (of the raw data, but also of transformations, such as moving averages),
and even basic modelling (such as regression). Data will be restricted to daily frequency.
The program should provide a comprehensive but easy to use text interface (i.e. no graphical interface),
based around a carefully designed structure of menus and sub-menus. It should be easy to use, and require
no programming knowledge from the end user.

Data Gathering

The frst task will be to load a list of active trading companies. There are several sources of such data. For
example, the NASDAQ stock exchange offers a direct CSV download:
http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchan0ge=nasdaq&render=download
You should also provide the ability to source data for any company. Trading data is widely available; see
for example:
https://www.quantshare.com/sa-43-10-ways-to-download-historical-stock-quotes-data-for-free

Description

The program should be able to provide descriptive analytics of the trading price for any company the user
selects. These include (but are not limited to):
1. Mean;
2. Quartiles;
3.Range;
4. Standard variation;
5. Coefcient of variation;

It should also provide a way to graphically visualise data related to any company, such as:
1. Raw time-series;
2. Trend lines;
3. Moving Averages (e.g. MA(n), with user-selectable n);
4. Weighted Moving Averages;
5. Moving Average Convergence/Divergence (MACD);


Prediction

The program should provide closing price predictions for companies specified by the user, with regression
models, using the following process (or similar):
1. The user specifies the modelling period (i.e. the training data);
2. They also specify a date for which they require a prediction;
3. A linear model is built, using the specified period;
4. The prediction is produced, along with the model's RMSE and R2 value (coefcient of determination).
Any other prediction approaches you may choose to implement (e.g. non-linear regression models) will add
credit to your project.
