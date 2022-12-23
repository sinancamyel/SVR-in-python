import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('maas.csv')
print(veriler)

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
X=scx.fit_transform(x.values)
scy=StandardScaler()
Y=scy.fit_transform(y.values)

from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(X,svr.predict(Y),color='blue')
plt.ylabel('Maaş')
plt.xlabel('Pozisyon Seviyesi')
plt.title('Çalışanların Pozisyonlarına Göre Maaşları')
plt.show()