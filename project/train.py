# train classifier
from sklearn.neighbors import KNeighborsClassifier
#train data 10 is mapped to prediction 1
#train data 100 is mapped to prediction 0
train_data = [[10], [100]]
train_target = [1, 0]
clf = KNeighborsClassifier(2)
clf.fit(train_data, train_target)

#save trained classifier as joblib file for server to use
import joblib
joblib.dump(clf, "binary_clf.joblib")