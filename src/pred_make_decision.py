import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Read data and do prediction
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)

# define the Y dataset with the date so that we can filter the last 3 months
Y = df[['Date', 'Close']]

# save the company name and category
com_cat = df[df['Date'] >= '2022-08-01'][['Company', 'Sector']]

index_df = df[['Date']]
df = df.drop(['Date'], axis=1)

Acc = []

# deal with the factor variables company and category
new_df = pd.get_dummies(df)

# define the X dataset
X = new_df.drop(['Close'], axis=1)

# scale the variables in X
scaler = MinMaxScaler()
sc_df_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# add the date again so that we can filter the last 3 months
sc_df_X = pd.concat([index_df, sc_df_X], axis=1)



# split the data of X and Y for train and test
train_X = sc_df_X[sc_df_X['Date'] < '2022-08-01']
train_Y = Y[Y['Date'] < '2022-08-01']['Close']
Train_date = train_X['Date']
train_X = train_X.drop(['Date'], axis=1)


test_X = sc_df_X[sc_df_X['Date'] >= '2022-08-01']
test_Y = Y[Y['Date'] >= '2022-08-01']['Close']
test_date = test_X['Date']
test_X = test_X.drop(['Date'], axis=1)

model_1 = LinearRegression()
model_1.fit(train_X, train_Y)

y_pred_1 = model_1.predict(test_X)

pred_df = pd.DataFrame({'Actual Close': test_Y, 'Predicted Close': y_pred_1})
pred_df.head()

pred_result = pd.concat([test_date,com_cat ,pred_df], axis=1)


print("Accuracy score of the predictions: {0}".format(r2_score(test_Y, y_pred_1)))
Acc.append(r2_score(test_Y, y_pred_1))

plt.figure(figsize=(8,8))
plt.ylabel('Close Price', fontsize=16)
plt.plot(pred_df)
plt.legend(['Actual Value', 'Predictions'])
plt.show()


com = pd.unique(pred_result['Company'])
date_close = pd.DataFrame(pd.unique(pred_result['Date']))

for company in com:
    com_df = pred_result[pred_result['Company'] == company].reset_index()['Predicted Close']
    date_close = pd.concat([date_close, com_df], axis = 1)

col_name = list(com)
col_name.insert(0,'Date')
date_close.columns = col_name
stock_price = date_close.set_index('Date')






# Determine the weight for each stock (how much should I invest in each stock)

# dataframe of increase rate per day for each company in the last 3 month
stock_return = stock_price.pct_change().dropna()


# Test the weights for each stock when dong the investment

# covariance matrix
cov_mat = stock_return.cov()
# covariance matrix for a year
cov_mat_year = cov_mat * 252

number = 10000

ran_p = np.empty((number, 53))

np.random.seed(53)

# randomly take 10000 weights for each stock and calculate the return and volatility for a year
for i in range(number):

    ran_51 = np.random.random(51)
    ran_weight = ran_51/np.sum(ran_51)

    mean_return = stock_return.mul(ran_weight,axis=1).sum(axis=1).mean()
    year_return = (1 + mean_return) ** 252 - 1
    random_volatility = np.sqrt(np.dot(ran_weight.T, np.dot(cov_mat_year, ran_weight)))

    ran_p[i][:51] = ran_weight
    ran_p[i][51] = year_return
    ran_p[i][52] = random_volatility


RandomPortfolios=pd.DataFrame(ran_p)


RandomPortfolios.columns=[company +'_weight' for company in list(com)]+['Returns','Volatility']

# plot the dataframe
RandomPortfolios.plot('Volatility','Returns',kind='scatter',alpha=0.3)
plt.show()

# calculate the sharpe ratio
RandomPortfolios['Sharpe'] = RandomPortfolios.Returns / RandomPortfolios.Volatility

# find the index that maximize the sharpe ratio and find the corresponding returns and volatility
max_index = RandomPortfolios.Sharpe.idxmax()
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[max_index,'Volatility']
y = RandomPortfolios.loc[max_index,'Returns']

# find the point in the plot assigned by red point
plt.scatter(x, y, color='red')
plt.text(np.round(x,3),np.round(y,3),(np.round(x,3),np.round(y,3)),ha='left',va='bottom',fontsize=10)
plt.show()

# find the corresponding weights for each stock
MSR_weights = np.array(RandomPortfolios.iloc[max_index, 0:51])
print(MSR_weights)
invest_com = pd.concat([pd.DataFrame(com), pd.DataFrame(MSR_weights)], axis=1)
invest_com.columns = ['Company', 'Weight']




# Use double moving average to decide when to sell and  buy
for company in com:
    date_com_stock = pd.DataFrame(stock_price[company])
    # calculate the average for 5 days and 15 days
    ma5 = date_com_stock[company].rolling(5).mean()
    ma15 = date_com_stock[company].rolling(15).mean()
    # plot these two average of 5 days and 15 days
    # plt.plot(ma5[15:])
    # plt.plot(ma15[15:])

    # remove the null value since the long moving average is 15 day
    ma5 = ma5[15:]
    ma15 = ma15[15:]
    date_com_stock = date_com_stock[15:]

    # find the death cross and golden cross
    s1 = ma5 < ma15
    s2 = ma5 > ma15
    death = s1 & s2.shift(1)
    death_date = date_com_stock.loc[death].index
    golden = ~(s1 | s2.shift())
    golden_date = date_com_stock.loc[golden].index

    print('For stock:', company)
    print('Golden cross:')
    for g in golden_date:
        print(g)
    print('Death cross:')
    for d in death_date:
        print(d)
    print('\n')




