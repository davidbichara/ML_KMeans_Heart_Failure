from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

heart = pd.read_csv('./heart.csv')



X = pd.DataFrame(heart,
                  columns = ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak'])
y = pd.DataFrame(heart,
                  columns = ['HeartDisease'])

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KMeans(n_clusters=2, random_state=1)

#model below does not include y because it is clustering solely off of X
model.fit(X_train)
predictions = model.predict(X_test)

labels = model.labels_
print("labels ", labels)
print("predictions ", predictions)
print("accuracy ", accuracy_score(y_test, predictions))