import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing

extra_feats = torch.load("extended_syn-data-500.pt")
extra_feats = sorted(extra_feats, key=lambda x: x[0])

mean = torch.zeros(5)
count = 0

for i in range(len(extra_feats)):
    count += extra_feats[i][1].shape[0]
    mean = mean + extra_feats[i][1].abs().sum(dim=0)

mean = mean / count

feats_normalized = []
for i in range(len(extra_feats)):
    feats_normalized.append(extra_feats[i][1].abs() - mean)

feats = [(i, feats_normalized[i]) for i in range(len(extra_feats))]

values = []
for i in range(len(feats)):
    tmp, _ = feats[i][1].max(dim=0) # max pooling
    #tmp = feats[i][1].mean(dim=0) # average pooling
    values.append(tmp.tolist())


X = np.array(values)

y = [0]*100 + [1]*100 + [2]*100 + [3]*100 + [4]*100


scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

res = []
res_five = []
for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)
    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    tmp = accuracy_score(y_test, y_pred)
    res.append(tmp)
    matrix = confusion_matrix(y_test, y_pred)
    res_five.append(matrix.diagonal()/matrix.sum(axis=1))

print("Accuracy: ")
print(np.array(res).mean())
print(np.array(res).std())

print("Accuracy for each class: ")
for num in range(5):
    results = []
    for i in range(20):
        temp = res_five[i][num]
        results.append(temp)
    print(np.array(results).mean())
    print(np.array(results).std())