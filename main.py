import alpaca_trade_api as tradeapi
import time
import pandas as pd
import sys
import schedule
import datetime
from pymongo import MongoClient
from analysis import analisis
from scraping import ikuzo

key = "PKFLYCVA7RFOAZGVBWPJ"
sec = "dA7gSxpkaBGacEbuwclt8p0KJzgGzg4QvLz9lkD9"


nama_db_price = 'hasil-analisis'
saham_list = [x.strip() for x in open('saham_list.txt', 'r').read().split('#')]
client = MongoClient('localhost', 27017)
db_price = client[nama_db_price]


url = "https://paper-api.alpaca.markets"
api = tradeapi.REST(key, sec, url, api_version='v2')
account = api.get_account()
position = api.list_positions()
balance_change = float(account.equity) - float(account.last_equity)
cbalance = float(account.equity)
orders = api.list_orders()
clock = api.get_clock()
isOpen = clock.is_open
dfo = pd.DataFrame()


#get_position = api.get_position()

#print(api.get_portfolio_history(date_start='2020-12-30'))
#print(len(saham_list))
#get_position = api.get_position('AAPL').unrealized_pl



#print(api.get_position('AAPL'))
#print(isOpen)
#print(api.list_orders())
#print(get_position)
rorders=[]
list_saham = {}
for saham in saham_list:
	#simbol = saham
	#list_saham[saham] = simbol
	print(saham)

def marketstat():
	if isOpen == False:
	    #clock = api.get_clock()
	    openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
	    currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
	    timeToOpen = int((openingTime - currTime) / 60)
	    print('MARKET CURRENTLY CLOSED')
	    print(str(timeToOpen) + " MINUTES REMAINING....")
	    #time.sleep(60)
	    sys.exit()
	else:
		pass




def b_s():

	marketstat()
	ikuzo()
	#time.sleep(5)
	analisis()
	hasil = []
	#print(account)
	list_dfs = {}
	for dfq in saham_list:
		try:
			get_positionz = api.get_position(dfq).unrealized_pl
			get_positionf = float(get_positionz)
			hasil.append(get_positionf)
		except:
			pass
		df = pd.DataFrame()
		list_dfs[dfq] = df
		df = pd.DataFrame(list(db_price[dfq].find()))
		df = df.tail(1)
		df['ALL_TRENDS'] = df['ALL_TRENDS'].abs()
		#df['Close'] = df['Close'].astype(str)
		kuantitas = df.iloc[0]['ALL_TRENDS']
		kuantitas = kuantitas.astype(str)
		#final_decision = df['FINAL_DECISION'].astype('string')
		
		#print(df)
		#print(dfq)
		#final_decision.dtypes

		if df.iloc[0]['FINAL_DECISION'] == 1:
			print('BUYING.....')
			c_order = api.submit_order(symbol=dfq,
			qty=kuantitas,
			side='buy',
			type='market',
			time_in_force='day')
			print(c_order)
			time.sleep(5)
			dbqty = c_order.qty
			dbcreatedat = c_order.created_at
			dbside = c_order.side
			dbsymbol = c_order.symbol
			get_position = api.get_position(dbsymbol).unrealized_pl
			dbprice = api.get_position(dbsymbol).current_price
			dbpreprice = api.get_position(dbsymbol).lastday_price
			#get_positionf = float(get_position)
			#hasil.append(get_positionf)
			holding = False
			#api.list_orders()
			print('BUYING '+ dfq + ' COMPLETED')
		elif df.iloc[0]['FINAL_DECISION'] == -1:
			print('SELLING.....')
			c_sell = api.submit_order(symbol=dfq,
			qty=kuantitas,
			side='sell',
			type='market',
			time_in_force='day')
			print(c_sell)
			time.sleep(5)
			dbqty = c_sell.qty
			dbcreatedat = c_sell.created_at
			dbside = c_sell.side
			dbsymbol = c_sell.symbol
			get_position = api.get_position(dbsymbol).unrealized_pl
			dbprice = api.get_position(dbsymbol).current_price
			dbpreprice = api.get_position(dbsymbol).lastday_price
			#get_positionf = float(get_position)
			#hasil.append(get_positionf)
			holding = False
			#api.list_orders()
			print('SELLING ' + dfq + ' COMPLETED')
		else:
			print('HOLDING ' + dfq + '...')
			holding = True
			pass

		

		#dbprofit = balance

		
		jam = str(clock.next_close)
		#nama_db = 'hasil-trade'
		db_hasil_trade = client['hasil-trade']
		db_profit = client['profit-trade']
		db_hasil_trade_jam = db_hasil_trade[jam]
		db_profit_total = db_profit[jam]
		#print(hasil)
		
		#print('%.3f' % hasil2)
		
		#print(c_order.qty)
		#db_hasil_trade_date = db_hasil_trade[jam]
		if holding == False:

			hasil_transaksi = {
			'Simbol' : dfq,
			'Kuantitas' : dbqty,
			'Waktu_Pembuatan' : dbcreatedat,
			'Side' : dbside,
			'Symbol_from_API' : dbsymbol,
			'Current_Symbol_Profit' : get_position,
			'Waktu_Lokal' : datetime.datetime.now(),
			'Current_Balance' : cbalance,
			'Current_Symbol_Price' : dbprice,
			'Previous_Day_Price' : dbpreprice,
			'Balance_Change' : balance_change
			}
			#print(hasil_transaksi)

			#hasil.append(hasil_transaksi)
			db_hasil_trade_jam.insert_one(hasil_transaksi)
		else:
			pass
	#print(hasil)
	hasil2 = sum(hasil)
	profit = {
	'Waktu_Lokal' : datetime.datetime.now(),
	'Trade_Symbol' : saham_list,
	'Balance_Change' : balance_change,
	'Current_Balance' : cbalance,
	'Unrealized Profit' : hasil2
	}
	db_profit_total.insert_one(profit)
	
	
	print('BALANCE CHANGE : ' ,balance_change)






b_s()
schedule.every(1).minutes.do(b_s)
while True:
	schedule.run_pending()
	time.sleep(5)

