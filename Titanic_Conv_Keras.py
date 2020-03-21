import numpy as np
import sklearn
import seaborn as sns
import pandas as pd
import sys
import matplotlib.pyplot as plt
import math
import os
import keras

file = 'titanic\\train.csv'
df = pd.read_csv(file).copy()

file_test = 'titanic\\test.csv'
df_test = pd.read_csv(file_test).copy()

file_rst = 'titanic\\gender_submission.csv'
df_rst = pd.read_csv(file_rst).copy()

# Change Sex to number
sex_to_num = {'male': 1,'female': 0}
df['Sex'] = df['Sex'].map(sex_to_num)
df_test['Sex'] = df_test['Sex'].map(sex_to_num)
# change embarked to number
emb_to_num = {'C': 0,'Q': 1,'S': 2}
df['Embarked'] = df['Embarked'].map(emb_to_num)
df_test['Embarked'] = df_test['Embarked'].map(emb_to_num)
# df.head()

# change title to number
title = df.Name.str.split(expand=True,)
title_test = df_test.Name.str.split(expand=True,)
df["Name"] = title[1]
df_test["Name"] = title_test[1]
#print(np.unique(df["Name"],return_counts=True))

# title_to_num = {'Billiard,':0, 'Capt.':1, 'Carlo,':2, 'Col.':3, 'Cruyssen,':4, 'Don.':5, 'Dr.':6,
#         'Gordon,':7, 'Impe,':8, 'Jonkheer.':9, 'Major.':10, 'Master.':11, 'Melkebeke,':12,
#         'Messemaeker,':13, 'Miss.':14, 'Mlle.':15, 'Mme.':16, 'Mr.':17, 'Mrs.':18, 'Ms.':19,
#         'Mulder,':20, 'Pelsmaeker,':21, 'Planke,':22, 'Rev.':23, 'Shawah,':24, 'Steen,':25,
#         'Velde,':26, 'Walle,':27, 'der':28, 'the':29, 'y':30}
title_to_num = {'Billiard,':0, 'Capt.':0, 'Carlo,':0, 'Col.':0, 'Cruyssen,':0, 'Don.':5, 'Dr.':1,
        'Gordon,':0, 'Impe,':0, 'Jonkheer.':0, 'Major.':0, 'Master.':2, 'Melkebeke,':0,
        'Messemaeker,':0, 'Miss.':3, 'Mlle.':0, 'Mme.':4, 'Mr.':5, 'Mrs.':4, 'Ms.':4,
        'Mulder,':0, 'Pelsmaeker,':0, 'Planke,':0, 'Rev.':6, 'Shawah,':0, 'Steen,':0,
        'Velde,':0, 'Walle,':0, 'der':0, 'the':0, 'y':0}
df['Name'] = df['Name'].map(title_to_num)
df_test['Name'] = df_test['Name'].map(title_to_num)
# split cabin column to number of cabins and deck
no_cabins = df["Cabin"].copy()
no_of_deck = df["Cabin"].copy()
for e, value in enumerate(df["Cabin"]):
    if isinstance(value, float): 
        #print(value.type())
        no_cabins[e] = 0
        no_of_deck[e] = 0
    else:
        #print(type(value))
        no_cabins[e] = (1.0 + str(value).count(" "))
        no_of_deck[e] = str(value)[0]

no_cabins_test = df_test["Cabin"].copy()
no_of_deck_test = df_test["Cabin"].copy()
for e, value in enumerate(df_test["Cabin"]):
    if isinstance(value, float): 
        #print(value.type())
        no_cabins_test[e] = 0
        no_of_deck_test[e] = 0
    else:
        #print(type(value))
        no_cabins_test[e] = (1.0 + str(value).count(" "))
        no_of_deck_test[e] = str(value)[0]       
    
deck_to_num = {0: 0,'A': 1,'B': 2,'C': 3,'D': 4,'E': 5,'F': 6,'G': 7,'T': 8}
df["Cabin"] = no_of_deck.map(deck_to_num)
df.rename(columns={'Cabin':'Deck'},inplace=True)
df = df.join(no_cabins, how='left')
df["Cabin"] = np.array(df["Cabin"], dtype=np.float32)    

df_test["Cabin"] = no_of_deck_test.map(deck_to_num)
df_test.rename(columns={'Cabin':'Deck'},inplace=True)
df_test = df_test.join(no_cabins_test, how='left')
df_test["Cabin"] = np.array(df_test["Cabin"], dtype=np.float32)  

# check if travelling alone
notAlone = df["Parch"].copy()
notAlone_test = df_test["Parch"].copy()

for i, value in enumerate(df['SibSp']):
    if value != 0 or df['Parch'][i] != 0: notAlone[i]=1
    else: notAlone[i]=0
    
for i, value in enumerate(df_test['SibSp']):
    if value != 0 or df_test['Parch'][i] != 0: notAlone_test[i]=1
    else: notAlone_test[i]=0

df.rename(columns={'Parch':'Parch0'},inplace=True)
df = df.join(notAlone, how='left')
df.rename(columns={'Parch':'notAlone'},inplace=True)

df_test.rename(columns={'Parch':'Parch0'},inplace=True)
df_test = df_test.join(notAlone_test, how='left')
df_test.rename(columns={'Parch':'notAlone'},inplace=True)

#split ticket number
tickets = df.Ticket.str.split(expand=True,)
ticketsPrefix = df["Ticket"].copy()
ticketsNumber = df["Ticket"].copy()
ticketsNumber[:] = 0.0
# tickets = tickets.T
for i, ticket in enumerate(tickets[0]):
    # print(i,' ',ticket)
    try:
        if int(ticket): ticketsPrefix[i] = 0
    except:
        ticketsPrefix[i] = ticket

for i, ticket in enumerate(tickets[2]):
    # print(i,' ',ticket)
    try:
        if int(ticket): ticketsNumber[i] = int(ticket)
    except:
        0;

for i, ticket in enumerate(tickets[1]):
    # print(i,' ',ticket)
    try:
        if int(ticket) and ticketsNumber[i]==0.0: ticketsNumber[i] = int(ticket)
    except:
        0;
        
for i, ticket in enumerate(tickets[0]):
    # print(i,' ',ticket)
    try:
        if int(ticket) and ticketsNumber[i]==0.0: ticketsNumber[i] = int(ticket)
    except:
        0;
        
tickets_test = df_test.Ticket.str.split(expand=True,)
ticketsPrefix_test = df_test["Ticket"].copy()
ticketsNumber_test = df_test["Ticket"].copy()
ticketsNumber_test[:] = 0.0
# tickets = tickets.T
for i, ticket in enumerate(tickets_test[0]):
    # print(i,' ',ticket)
    try:
        if int(ticket): ticketsPrefix_test[i] = 0
    except:
        ticketsPrefix_test[i] = ticket

for i, ticket in enumerate(tickets_test[2]):
    # print(i,' ',ticket)
    try:
        if int(ticket): ticketsNumber_test[i] = int(ticket)
    except:
        0;

for i, ticket in enumerate(tickets_test[1]):
    # print(i,' ',ticket)
    try:
        if int(ticket) and ticketsNumber_test[i]==0.0: ticketsNumber_test[i] = int(ticket)
    except:
        0;
        
for i, ticket in enumerate(tickets_test[0]):
    # print(i,' ',ticket)
    try:
        if int(ticket) and ticketsNumber_test[i]==0.0: ticketsNumber_test[i] = int(ticket)
    except:
        0;
        
prefix_to_num = {'0':0, 'A./5.':1, 'A.5.':1, 'A/4':2, 'A/4.':2, 'A/5':1, 'A/5.':1, 'A/S':1, 'A4.':2,
        'C':3, 'C.A.':4, 'C.A./SOTON':4, 'CA':4, 'CA.':4, 'F.C.':5, 'F.C.C.':5, 'Fa':6,
        'LINE':7, 'P/PP':8, 'PC':9, 'PP':8, 'S.C./A.4.':9, 'S.C./PARIS':10, 'S.O./P.P.':10,
        'S.O.C.':9, 'S.O.P.':10, 'S.P.':10, 'S.W./PP':10, 'SC':9, 'SC/AH':11, 'SC/PARIS':10,
        'SC/Paris':10, 'SCO/W':12, 'SO/C':9, 'SOTON/O.Q.':13, 'SOTON/O2':13, 'SOTON/OQ':13,
        'STON/O':13, 'STON/O2.':13, 'SW/PP':14, 'W./C.':15, 'W.E.P.':16, 'W/C':15, 'WE/P':16}
df['Ticket'] = ticketsPrefix.map(prefix_to_num)
df.rename(columns={'Ticket':'TicketPrefix'},inplace=True)    
df = df.join(ticketsNumber, how='left')
df.rename(columns={'Ticket':'TicketNumber'},inplace=True)  
df["TicketNumber"] = np.array(df["TicketNumber"], dtype=np.float32)   
#df = df.drop(columns={'Ticket', 'PassengerId'})
#df.head()
df = df.drop(columns={'PassengerId'})

df_test['Ticket'] = ticketsPrefix_test.map(prefix_to_num)
df_test.rename(columns={'Ticket':'TicketPrefix'},inplace=True)    
df_test = df_test.join(ticketsNumber_test, how='left')
df_test.rename(columns={'Ticket':'TicketNumber'},inplace=True)  
df_test["TicketNumber"] = np.array(df_test["TicketNumber"], dtype=np.float32)   
#df = df.drop(columns={'Ticket', 'PassengerId'})
#df.head()
df_test = df_test.drop(columns={'PassengerId'})

# col_numerical = list(df.describe().columns)
# col_cat = list(set(df.columns).difference(col_numerical))
# remove_list = []
# col_categorical = [e for e in col_cat if e not in remove_list]

#replace NaN Age with mean value (depended of sex)
AgeM = []
AgeF = []
for i, agepass in enumerate(df['Age']): 
    if np.isnan(agepass) == False: 
        if df['Sex'][i] == 1: AgeM.append(agepass)
        elif df['Sex'][i] == 0: AgeF.append(agepass)

meanAgeM = np.mean(AgeM)
meanAgeF = np.mean(AgeF)

for i, age in enumerate(df['Age']):
    if np.isnan(age) and  df['Sex'][i] == 1: age = meanAgeM  
    elif np.isnan(age) and  df['Sex'][i] == 0: age = meanAgeF     
AgeM_test = []
AgeF_test = []
for i, agepass in enumerate(df_test['Age']): 
    if np.isnan(agepass) == False: 
        if df_test['Sex'][i] == 1: AgeM_test.append(agepass)
        elif df_test['Sex'][i] == 0: AgeF_test.append(agepass)

meanAgeM_test = np.mean(AgeM_test)
meanAgeF_test = np.mean(AgeF_test)

for i, age in enumerate(df_test['Age']):
    if np.isnan(age) and  df_test['Sex'][i] == 1: age = meanAgeM_test  
    elif np.isnan(age) and  df_test['Sex'][i] == 0: age = meanAgeF_test 
    
# replane NaN with 0; Next idea is to replace with average value
df[:].fillna(0, inplace=True)
df_test[:].fillna(0, inplace=True)

array = df.values
X=array[:,1:]
y=array[:,0]

mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std

array_test = df_test.values
X_test=array_test[:,:]
# y=array[]

X_test -= mean
X_test /= std
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

# X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1)
# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X_train, y_train)
# print(accuracy_score(y_train, clf.predict(X_train)))
# print(classification_report(y_train, clf.predict(X_train)))
# print(confusion_matrix(y_train, clf.predict(X_train)))

# print(accuracy_score(y_validation, clf.predict(X_validation)))
# print(classification_report(y_validation, clf.predict(X_validation)))
# print(confusion_matrix(y_validation, clf.predict(X_validation)))

from keras import layers
from keras import models

def build_model():
    model = models.Sequential()
    # model.add(layers.LSTM(1024))
    # model.add(layers.LSTM(512))
    model.add(layers.Conv1D(1024, 3,activation='relu'))
    model.add(layers.MaxPooling1D((3)))
    model.add(layers.Flatten())
    # model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.2))
    # # model.add(layers.Dense(512, activation='tanh'))
    # # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(64, activation='tanh'))
    # model.add(layers.Dense(1, activation='sigmoid'))
    #model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'],
              )
    return model
from keras import optimizers



from keras import backend as K
k = 5
num_val_samples = len(X) // k
# Some memory clean-up
K.clear_session()
num_epochs = 61
acc_histories = []
val_acc_histories = []
loss_histories = []
val_loss_histories = []
# poczatek komentarza
for i in range(k):
    print('processing fold #', i)
    # Przygotowuje dane walidacyjne: dane z k-tej składowej.
    val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]

    # Przygotowuje dane treningowe: dane z pozostałych składowych.
    partial_train_data = np.concatenate(
        [X[:i * num_val_samples],
          X[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y[:i * num_val_samples],
          y[(i + 1) * num_val_samples:]],
        axis=0)

    # Buduje model Keras (model został skompilowany wcześniej).
    model = build_model()
    # Przeprowadza ewaluację modelu przy użyciu danych walidacyjnych.
    partial_train_data = np.expand_dims(partial_train_data, 2)
    val_data = np.expand_dims(val_data, 2)
    history = model.fit(partial_train_data, 
                        partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, 
                        batch_size=512, 
                        verbose=0
                        )
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc_histories.append(acc)
    val_acc_histories.append(val_acc)
    loss_histories.append(loss)
    val_loss_histories.append(val_loss)
   #koniec komentarza 
# history = model.fit(X_train, 
#                     y_train, 
#                     epochs=num_epochs, 
#                     batch_size=512,
#                     validation_data=(X_validation,y_validation))
#test_loss, test_acc = model.evaluate(X_validation, y_validation)


average_val_acc_history = [np.mean([x[i] for x in val_acc_histories]) for i in range(num_epochs)]
average_acc_history = [np.mean([x[i] for x in acc_histories]) for i in range(num_epochs)]
import matplotlib.pyplot as plt
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

plt.plot(average_val_acc_history)
plt.plot(average_acc_history)
plt.figure()

# plt.plot(num_epochs, loss_histories)
# plt.plot(num_epochs, val_loss_histories)
plt.show()
# print(test_acc)

model = build_model()
X = np.expand_dims(X, 2)

history = model.fit(X, 
                    y,
                    # validation_data=(val_data, val_targets),
                    epochs=num_epochs, 
                    batch_size=512, 
                    verbose=0
                    )
# ts=62
result0 = df_rst["Survived"].copy()
result = result0.copy()

X_predict = np.array(X_test)

results = model.predict_classes(np.expand_dims(X_test, 2))
results = results.T

result[:]=results[:][0]
df_rst.rename(columns={'Survived':'Sur'},inplace=True)
df_rst = df_rst.join(result, how='left')

df_rst_f = df_rst.copy()

df_rst_f = df_rst_f[df_test.Sex != 1]


print(accuracy_score(df_rst_f["Sur"], df_rst_f["Survived"]))
print(classification_report(df_rst_f["Sur"], df_rst_f["Survived"]))
print(confusion_matrix(df_rst_f["Sur"], df_rst_f["Survived"]))

df_rst = df_rst.drop(columns={'Sur'})

df_rst.to_csv('submission.csv', index = False)