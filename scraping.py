import requests
from bs4 import BeautifulSoup as bs4
import os
import numpy as np
import sys
import time
import schedule
import csv
import pandas as pd
import pymongo
from pymongo import MongoClient
import datetime



saham_list = [x.strip() for x in open("saham_list.txt", "r").read().split('#')]
headers = { 'User-Agent': 'Mozilla/5.0',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': 'https://www.investing.com/',
            'X-Requested-With': 'XMLHttpRequest'}
payload = {'pairID':101325,'period':'','viewType':'normal'}
payloadh = {'header': 'BBCA Historical Data', 'st_date': '12/01/2018', 'end_date': '12/01/2020', 'sort_col': 'date',
 'action': 'historical_data', 'smlID': '1450174', 'sort_ord': 'DESC', 'interval_sec': 'Daily', 'curr_id': '101354'}
periods = {'1 Menit':60} #,'5 Menit':300 #,'15 Menit':900, '30 Menit':1800, '1 Jam':3600}
#pairIDs = {'BBCA':101354, 'Mandiri':101325}
t = time.localtime()
timestr = time.strftime("%H.%M|%d-%m-%Y")
tanggal = time.strftime('%d-%m-%Y')

def ikuzo():
	print("Jumlah Saham : ", len(saham_list))
	print("Waktu : ", str(timestr))

	for saham in saham_list:
		for k,v in periods.items():
			payload['period'] = v
			tt = time.strftime('%H.%M.%S')
			print('Period : ',k)
			print('Simbol : ',saham)
			print('Waktu : ', tt)
			if saham == 'AAPL':
				url_saham = 'apple-computer-inc'
				#n = "C:/Users/User/Desktop/TA BUAT GITHUB BOY/Technical2.txt"
				#nn = "C:/Users/User/Desktop/TA BUAT GITHUB BOY/Historical2.csv"
				nnn = url_saham + '/Summary2.csv'
			elif saham == 'MSFT':
				url_saham = 'microsoft-corp'
				nnn = url_saham + '/Summary2.csv'
			elif saham == 'INTC':
				url_saham = 'intel-corp'
				nnn = url_saham + '/Summary2.csv'
			elif saham == 'NKE':
				url_saham = 'nike'
				nnn = url_saham + '/Summary2.csv'
			elif saham == 'IBM':
				url_saham = 'ibm'
				nnn = url_saham + '/Summary2.csv'
			elif saham == 'NVDA':
				url_saham = 'nvidia-corp'
				nnn = url_saham + '/Summary2.csv'
			elif saham == 'ATVI':
				url_saham = 'activision-inc'
				nnn = url_saham + '/Summary2.csv'
			elif saham == 'FB':
				url_saham = 'facebook-inc'
				nnn = url_saham + '/Summary2.csv'
			else:
				print('UNDER DEVELOPMENT...')
				sys.exit()

			urlo = 'https://www.investing.com/equities/'+ url_saham
			urlt = 'https://www.investing.com/equities/'+ url_saham +'-technical'
			urlh = 'https://www.investing.com/instruments/HistoricalDataAjax'

			reqo = requests.post(urlo, headers = headers, data = payload)
			reqt = requests.post(urlt, headers = headers, data = payload)
			reqh = requests.post(urlh, headers = headers, data = payloadh)

			soupo = bs4(reqo.content, 'lxml')
			soupt = bs4(reqt.content, 'lxml')
			souph = bs4(reqh.content, 'lxml')

			osumm = soupo.find('div', class_='instrument-price_instrument-price__3uw25 instrument-price_instrument-price-lg__3ES-Q')
			osumma = osumm.find_all('span')
			gsumm = soupo.find('ul', class_='trading-hours_trading-hours__3q1Yp mt-3')
			gsumma = gsumm.find_all('li')
			bsumm = soupo.find('div', class_='instrument-page_section__79xMl instrument-page_border__ufzK4')
			bsumma = bsumm.find_all('span')

			tTA = soupt.find('table', id="curr_table", class_='genTbl closedTbl technicalIndicatorsTbl smallTbl float_lang_base_1')
			tMA = soupt.find('table', id='curr_table', class_='genTbl closedTbl movingAvgsTbl float_lang_base_2')
			srTA = tTA.find_all('tr') #HEADER
			ssTA = tTA.find_all('p') #CONTENT
			srMA = tMA.find_all('tr')
			ssMA = tMA.find_all('p')

			hist = souph.find('table', id='curr_table')
			histo = hist.find_all('tr')
			#print(osumma)




			os.makedirs(os.path.dirname(nnn), exist_ok=True)
			#of = open(n, "w")
			#ofc = open(nn, "w")
			ofco = open(nnn, 'a', newline='')
			headercsv = ['Price', 'Percent', 'Change', 'Bid', 'Ask', 'Low', 'High', 'Open', 'Symbol', 'Volume', 'Date']
			ofcou = csv.writer(ofco, lineterminator = '\n')


			mydata = []
			a = csv.writer(ofco, delimiter=',')


			hlTA = srTA[:1]
			srrTA = srTA[1:-1]
			hlMA = srMA[:1]
			srrMA = srMA[1:-1]

			histor = histo[0:1]
			histor2 = histo[:0:-1]


			### CHANGE CSV ORDER ###
			osummar = osumma[:1] #PRICE
			osummar2 = osumma[1:-1] #CHANGE
			osummar3 = osumma[:1:-1] #Percent

			gsummar = gsumma[1:2:1] #BID/ASK
			gsummar2 = gsumma[-1:1:-1] #Day's Range
			gsummar3 = gsumma[:1] #VOLUME

			bsummar = bsumma[14:16]
			#print(bsummar)
			#ofcou.writerow(h for h in headercsv) 

			realtime = []
			mdb = []

			for row in osummar:
				y = list(row.stripped_strings)
				y = [x.replace(',','')for x in y]
				#print(y)
				if len(y) > 0:
					new_dat = y[0]
					mdbprice = y[0]
					mydata.append(new_dat)

			for row in osummar2:
				y = list(row.stripped_strings)
				y = [x.replace(',','')for x in y]
				if len(y) > 0:
					try:
						new_dat = y[1]
						for row in osummar3:
							y = list(row.stripped_strings)
							y = [x.replace(',','')for x in y]
						if len(y) > 0:
							new_datz = y[2] + '%'
							mydata.append(new_datz)
					except:
						new_dat = y[0]
						for row in osummar3:
							y = list(row.stripped_strings)
							y = [x.replace(',','')for x in y]
						if len(y) > 0:
							new_datz = y[1] + '%'
							mydata.append(new_datz)
					
					mydata.append(new_dat)
			#print(mydata)

			'''for row in osummar3:
				y = list(row.stripped_strings)
				y = [x.replace(',','')for x in y]
				if len(y) > 0:
					new_dat = y[2] + '%'
					mydata.append(new_dat)'''

			for row in gsummar:
				y = list(row.stripped_strings)
				y = [x.replace(',','')for x in y]
				if len(y) > 0:
					new_dat = y[2]
					new_dat2 = y[4]
					mydata.append(new_dat)
					mydata.append(new_dat2)

			for row in gsummar2:
				y = list(row.stripped_strings)
				y = [x.replace(',','')for x in y]
				if len(y) > 0:
					new_dat = y[2]
					new_dat2 = new_dat.split('-')
					new_dats = new_dat2[0]
					new_dats2 = new_dat2[1]
					mdblow = new_dat2[0]
					mdbhigh = new_dat2[1]

					mydata.append(new_dats)
					mydata.append(new_dats2)

			for row in bsummar:
				y = list(row.stripped_strings)
				y = [x.replace(',','')for x in y]
				if len(y) > 0:
					new_dat = y[0]
					mdbopen = y[0]
					symb = saham

					mydata.append(new_dat)
					mydata.append(symb)


			for row in gsummar3:
				y = list(row.stripped_strings)
				y = [x.replace(',','')for x in y]
				if len(y) > 0:
					new_dat = y[2]
					mdbvol = y[2]
					new_dat2 = tanggal
					

					mydata.append(new_dat)
					mydata.append(new_dat2)

			connection = MongoClient('localhost',27017)
			db = connection['symbol-data']
			names = db[symb]

			mdbprice = float(mdbprice)
			mdblow = float(mdblow)
			mdbhigh = float(mdbhigh)
			mdbopen = float(mdbopen)
			mdbvol = float(mdbvol)

			date = datetime.datetime.now()



			datamdb = {
			'_id' : date,
			'open' : mdbopen,
			'high' : mdbhigh,
			'low' : mdblow,
			'close' : mdbprice,
			'volume' : mdbvol
		    }

			names.insert_one(datamdb)
			a.writerow(i for i in mydata)

			#print(datamdb)






ikuzo()
