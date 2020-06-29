import quandl,datetime,math
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


style.use('ggplot')
quandl.ApiConfig.api_key = "r6DcL_rTRMXzwxzjFkru"   #Quandl bizi onaylıyor ve veri çekmemzie izin veriyor.

df = quandl.get("BITFINEX/BTCEUR")                     # quandldan verileri çekerek dataframe oluşturuyoruz


df['HL_PCT'] = (df['High']-df['Low']) / df['Last'] * 100.0                #2 parametre oluşturduk
df['ASK-BID_PCT'] = (df['Ask'] - df['Bid']) /df['Ask'] * 100.0


df = df[['High','Low','Volume','Last','HL_PCT','ASK-BID_PCT']]             #yeni oluşturduğumuz parametrelerle diğer parametreleri aldık


forecast_out = int(math.ceil(len(df) * 0.01))                               #dataframein yüzde 1ini alıyoruz onu bir üst sayıya yuvarlıyoruz.
forecast_col = 'Last'

df['Label'] = df[forecast_col].shift(-forecast_out)                        #tahmin edilecek olan sütunu oluşturduk yani label. aslında bu sütun paranın son değerini oluşturuyor.

X = df.iloc[:,:-1]
y = df.iloc[:,-1]


X_lately = X[-forecast_out:]
X = X[:-forecast_out]
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)  #x ve y 4 parçaya bölündü. x ve ye 2şer parçaya  bölündü bunu yapmamızın sebebi tüm veriyi bilgisayara train ettirmek için kullanmayacağız
                                                                                            #belirli bir kısmını error hesabı için kullanacağız ki yazılımış olan kod ne kadar düzgün çalışıyor görmek için.
                                                                                            #train size x ve y nin yüzde 80ini algoritmayı geliştirmek için geri kalan 20si de error hesabı yani test için kullanılıyor.



regressor = LinearRegression()               #algoritmamızı oluşturduk ve sonra
regressor.fit(X_train,y_train)               #algoritmamızın üzerine train ettiğimiz verileri yerleştiriyoruz.

accuracy = regressor.score(X_test,y_test)   #algoritmamızın ne kadar doğru çalıştığını göreceğiz yani aslında error hesabı yapacağız.

print(accuracy)
forecast_set = regressor.predict(X_lately)
df['Forecast']  = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for i in range(len(df.columns)-1) ] + [i]

df['Last'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price(USD)')
plt.legend(loc=4)
plt.show()











