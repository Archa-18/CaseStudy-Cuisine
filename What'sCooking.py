import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm

rf=RandomForestClassifier(random_state=1)
nb=MultinomialNB()
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()

d_c = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian',
 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole',
 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']

data = pd.read_json('train.json')

#print(data['cuisine'].unique())
x=data['ingredients']
y=data['cuisine'].apply(d_c.index)
data['all_ingredients']=data['ingredients'].map(';'.join)

cv=CountVectorizer()
x= cv.fit_transform(data['all_ingredients'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

rf.fit(x_train, y_train)
nb.fit(x_train, y_train)
dt.fit(x_train, y_train)
sv.fit(x_train, y_train)

rf_predict = rf.predict(x_test)
nb_predict = nb.predict(x_test)
dt_predict = dt.predict(x_test)
sv_predict = sv.predict(x_test)

print('RandomForest', accuracy_score(y_test, rf_predict))
print('NaiveBayes', accuracy_score(y_test, nb_predict))
print('DecisionTreeClassifier', accuracy_score(y_test, dt_predict))
print('svm', accuracy_score(y_test, sv_predict))

#RandomForest 0.7583909490886235
#NaiveBayes 0.7323695788812068
#DecisionTreeClassifier 0.6419861722187303
#svm 0.7859208045254557