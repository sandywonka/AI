import sys
import datetime
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
plt.rcParams['figure.figsize'] = 15,7
plt.rcParams['font.size'] = 8
#plt.style.use('ggplot')
plt.style.use('default')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import tree

import pymongo
from pymongo import MongoClient

from nowtrade.data_connection import MongoDatabaseConnection
#from tesjualbeli import get_orders, order


def analisis():


    nama_db = 'symbol-data'
    nama_db_price = 'hasil-analisis'
    nama_saham = [x.strip() for x in open('saham_list.txt', 'r').read().split('#')]
    #nama_saham = 'NVDA'


    client = MongoClient('localhost', 27017)
    db = client[nama_db]
    db_price = client[nama_db_price]



    mdb_hasil_analisis = []
    for dfq in nama_saham:
        df = pd.DataFrame()
        df = pd.DataFrame(list(db[dfq].find()))
        mdb_df = df
        mdb_df['INSERT_TIME'] = datetime.datetime.now()
        df = df.drop(columns=['INSERT_TIME'])
        





        df.rename(columns={'open': 'Open',
                                    'high': 'High',
                                    'low': 'Low',
                                    'close': 'Close',
                                    'volume': 'Volume',
                                    'adj_close': 'Adj Close',
                                    '_id': 'Date'},
                                   inplace=True)
        
        df['Symbol'] = dfq
        df['Symbol'] = df['Symbol'].astype('string')
        df['change_in_price'] = df['Close'].diff()
        hargas = df['Close']

        df_date = df['Date']
        df = df.set_index('Date')
        #df.sort_values(by = ['Symbol','Date'], inplace = True)
        price_data = df[df['Volume'] != 0]
        price_data.tail()
        #trend = 0
        print('\nSIMBOL : ',dfq)



        #price_data = pd.read_csv(df)


        


        ### RSI ###

        n = 8


        up_df, down_df = price_data[['change_in_price']].copy(), price_data[['change_in_price']].copy()
        up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
        down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
        down_df['change_in_price'] = down_df['change_in_price'].abs()
        ewma_up = up_df['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        ewma_down = down_df['change_in_price'].transform(lambda x: x.ewm(span = n).mean())

        relative_strength = ewma_up / ewma_down
        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))



        price_data['down_days'] = down_df['change_in_price']
        price_data['up_days'] = up_df['change_in_price']
        price_data['RSI'] = relative_strength_index
        threshold_up = 80 #REVISI###################
        threshold_down = 20

        ##########


        ### STOCHASTIC ###

        n = 8


        low_4, high_4 = price_data[['Low']].copy(), price_data[['High']].copy()
        low_4 = low_4['Low'].transform(lambda x: x.rolling(window = n).min())
        high_4 = high_4['High'].transform(lambda x: x.rolling(window = n).max())


        k_percent = 100 * ((price_data['Close'] - low_4) / (high_4 - low_4))


        price_data['Low_Sto'] = low_4
        price_data['High_Sto'] = high_4
        price_data['K_percent'] = k_percent

        ##########


        ### WILLIAMS %R ###

        n = 8


        low_8, high_8 = price_data[['Low']].copy(), price_data[['High']].copy()


        low_8 = low_8['Low'].transform(lambda x: x.rolling(window = n).min())
        high_8 = high_8['High'].transform(lambda x: x.rolling(window = n).max())

        r_percent = ((high_8 - price_data['Close']) / (high_8 - low_8)) * - 100


        price_data['R_percent'] = r_percent
        threshold_up_r = -20
        threshold_down_r = -80


        ###########


        ### MACD ###
        n = 8

        ema_26 = price_data['Close'].transform(lambda x: x.ewm(span = 26).mean())
        ema_12 = price_data['Close'].transform(lambda x: x.ewm(span = 12).mean())
        macd = ema_12 - ema_26


        ema_8_macd = macd.ewm(span = n).mean()


        price_data['MACD'] = macd
        price_data['MACD_EMA'] = ema_8_macd

        price_data['MACD_DIFF'] = price_data['MACD'] - price_data['MACD'].shift()
        threshold_up_macd = 0.07
        threshold_down_macd = -0.07




        ##########









        closing = price_data['Close']
        #data = closing.drop(closing.tail(1).index)
        #second_newest = closing.iloc[:2]
        #newest = closing.iloc[:1]
        #days_out = 30
        #price_data["label"] = [1 if x > newest else -1 if x < newest else 0] 
        closing = closing.transform(lambda x : np.sign(x.diff()))

        #closing = closing.transform(lambda x : x.shift(1) < x)

        price_data['Prediction'] = closing




        price_data['RSI_Trend'] = [ 1 if x >= threshold_up else -1 if x < threshold_down else 0 for x in price_data['RSI']]

        price_data['STO_Trend'] = [ 1 if x >= threshold_up else -1 if x < threshold_down else 0 for x in price_data['K_percent']]

        price_data['W%R_Trend'] = [ 1 if x >= threshold_up_r else -1 if x < threshold_down_r else 0 for x in price_data['R_percent']]

        price_data['MACD_Trend'] = [ 1 if x >= threshold_up_macd else -1 if x < threshold_down_macd else 0 for x in price_data['MACD_DIFF']]

        price_data['ALL_TRENDS'] = price_data['RSI_Trend'] + price_data['STO_Trend'] + price_data['W%R_Trend'] + price_data['MACD_Trend']







        #price_data.dtypes
        price_data['DECISION'] = [ 1 if x > 1 else -1 if x < -1 else 0 for x in price_data['ALL_TRENDS']]





        raee = price_data['DECISION']
        raeee = raee.iloc[-1]






        price_data = price_data.dropna()












        #######################################     RANDOM FOREST     ##################################

        X_Cols = price_data[['RSI', 'MACD', 'K_percent', 'R_percent', 'MACD_DIFF', 'MACD_EMA', 'Open', 'High', 'Low', 'Close', 'Volume']]
        L = X_Cols
        print(L.shape)
        #P = price_data
        Y_Cols = price_data['Prediction']
        #price_data['RF'] = ''

        #train_size = int(X_Cols.shape[0]*0.7)
        #X_train = X_Cols[:train_size]
        #X_test = X_Cols[train_size:]
        #y_train = Y_Cols[:train_size]
        #y_test = Y_Cols[train_size:]


        X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, train_size = 0.6)




        model = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 5, bootstrap = True)

        model.fit(X_train, y_train)

        x_pred_test = model.predict(X_test)
        x_pred_train = model.predict(X_train)
        y_pred_test = y_test
        y_pred_train = y_train


        df1 = pd.DataFrame(x_pred_train[:, None], columns = ['RF_Train']) #index=df_date.index)
        df2 = pd.DataFrame(x_pred_test[:, None], columns = ['RF_Train'])


        dff = pd.concat([df1, df2]).reset_index()
        dffx = dff.join(df_date)
        dffx = dffx.set_index('Date')
        price_data = price_data.join(dffx)
        #price_data = price_data.dropna()

        df3 = price_data['RF_Train'].isnull().sum()
        #price_data = price_data.fillna(9)
        #price_data = price_data.reset_index()
        price_data = price_data.shift(df3)
        price_data = price_data.drop(price_data.iloc[0:df3].index)
        #price_data = price_data.set_index('Date')

        #print(-df3)
        #df3 = df3.iloc[:,[-1]].reset_index()
        #df3 = df3.drop(columns=['Date'])

        #price_data = price_data.dropna()
        #df3x = pd.concat([df3, price_data['RF_Train']])
        #df3x = df3x.reset_index()
        #df3x = df3x.join(df_date)
        #df3x = df3x.drop(columns=['index'])
        #df3x = df3x.set_index('Date')
        #df3x = df3x.sort_index()
        #df3x = df3x.drop(columns=['RF_Train'])
        #df3x = df3x.rename(columns={0:'RF_Train'})
        #price_data = price_data.drop(columns=['RF_Train'])

        #price_data = price_data.join(df3x)

        #print(df3x)


        #out = price_data.values[np.argsort(df3, axis=0), np.arange(df3.shape[1])]
        #price_data = pd.DataFrame(out, columns=price_data.columns)
        



        #df3 = price_data.loc[price_data['RF_Train'].isnull()]
        #df3 = df3.iloc[:,[-1]].reset_index()
        #df3 = df3.drop(columns=['Date'])
        #df3x = pd.concat([price_data['RF_Train'], df3])
        #price_data = df3x.join(df_date)

        #print(price_data)
        #price_data['RF_Train'] = price_data.sort_values('Date', na_position='first')

        dfz = pd.concat([y_pred_train, y_pred_test])
        dfz = dfz.sort_index()
        price_data['RF_Test'] = dfz
        
        #############################################################################################





        ##############################################     AKURASI MODEL     ############################################
        print('Akurasi Model Saat Ini (%): ', accuracy_score(y_test, model.predict(X_test)) * 100.0)
        x_predz = model.predict(X_test)
        #print(x_predz)
        #################################################################################################################









        pred = x_pred_test




        #####################################     HASIL PREDIKSI PLOT     ###############################
        
        #plt.style.use('ggplot')
        #plt.plot(np.arange(len(pred)), pred, label='pred')
        #plt.plot(np.arange(len(y_test)), y_test, label='real' )
        #plt.title('Hasil Prediksi')
        
        #################################################################################################
        
        #########################################     SCORE     #########################################

        scores = cross_val_score(model, X_train, y_train, cv=10, scoring = "accuracy")
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard Deviation:", scores.std())
        #importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model.feature_importances_,3)})
        #importances = importances.sort_values('importance',ascending=False).set_index('feature')
        #importances.head(15)
        #importances.plot.bar()
        
        ################################################################################################


        feature_imp = pd.Series(model.feature_importances_, index=X_Cols.columns).sort_values(ascending=False)
        print(feature_imp)




        ###############################################################################################

        
        
        ######################################     TREE MODEL GRAPH     ################################
        feature_names = list(X_Cols)
        class_name = list(y_test)
        class_name = bool(class_name)


        estimator = model.estimators_[1]
        estimator2 = model.estimators_[1]
        ##model.estimators_[0].tree_.max_depth


        plt.style.use('default')
        fig = plt.figure(figsize=(25,20))
        
        LP = tree.plot_tree(estimator, feature_names = feature_names, class_names= class_name,  filled=True, rounded=True)
        plt.savefig('POHON_1 '+ dfq +'.png', dpi=150)
        
        #LP2 = tree.plot_tree(estimator2, feature_names = feature_names, filled = True)
        #plt.savefig('POHON_2 '+dfq+'.png', dpi=150)

        #plt.show()
        #plt.savefig('foo.png')
        
        
        #################################################################################################







        #plt.show()

        price_data = price_data.drop(columns = ['index'])
        price_data.to_csv('summaries '+ dfq +'.csv')
        dfz.to_csv('summaries_RF '+ dfq +'.csv')
        #print(price_data)
        
        

        #############################    INSERT MONGO HASIL ANALISIS     #########################
        
        time_now = datetime.datetime.now()
        db_price_name = db_price[dfq]
        #db_price_name.insert_one(data_mdb)
        
        price_data['RECORDS_TIME'] = datetime.datetime.now()
        price_data = price_data.tail(1)
        price_data = price_data.reset_index()
        #print(price_data.tail())
        db_price_name.insert_many(price_data.to_dict('records'))
        
        ##########################################################################################
        
        
        #dfz.head()
analisis()

