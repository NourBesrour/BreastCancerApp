import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

data = pd.read_csv("breast-cancer-dataset.csv")

data = data[~data.apply(lambda row: row.astype(str).str.contains('#').any(), axis=1)]
data.rename(columns = {'Tumor Size (cm)':'TS'}, inplace = True)
data.rename(columns = {'Breast Quadrant':'BQ'}, inplace = True)

print(data.info())
print(data.isnull().sum()) # there is no missing value
print(data.shape)
print(data.dtypes)

data['Age'] = data['Age'].astype(int)
data["TS"] = data["TS"].astype(float).astype(int)
print(data.dtypes)

def label_encode_object_columns(df):
    object_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in object_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    return df, label_encoders

# corr we find out that S/N ans Year raw data don't have any contribution to the model
data = data.drop('S/N', axis=1)
data = data.drop('Year', axis=1)
# Assuming your data is stored in a DataFrame called df
data, label_encoders = label_encode_object_columns(data.copy())
print(data.dtypes)

x, y = data.drop(columns='Diagnosis Result', axis=1), data['Diagnosis Result']
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=100)

# corr = data.corr()
# plt.figure(figsize=(14,9))
#sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.savefig('heatmap.png')
# plt.show()

def fit_model_metrics(x):
    x.fit(x_train, y_train)
    yhat = x.predict(x_test)
    y_known = x.predict(x_train)
    algoname = x.__class__.__name__
    accuracy = round(accuracy_score(y_test, yhat), 3)
    accuracy_train = round(accuracy_score(y_train, y_known), 3)
    precision = round(precision_score(y_test, yhat), 2)
    recall = round(recall_score(y_test, yhat), 2)
    f1 = round(f1_score(y_test, yhat), 2)
    return algoname, accuracy, accuracy_train, precision, recall, f1


column_names = ['Model', 'Accuracy', 'Accuracy on Train', 'Precision', 'Recall', 'F1 Score']
score = []
LR = LogisticRegression(max_iter=100, penalty='l1', solver='liblinear')
score.append(fit_model_metrics(LR))
print(pd.DataFrame(score, columns=column_names))





