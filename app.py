import  pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Определяем входные параметры модели
class IrisImput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

# Загружаем датасет
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# записываем датасет в CSV
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df.to_csv('data/datasets/iris_dataset.csv', index=False)

# выполним предварительную обработку данных
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaler, y_train)

# записываем модель в фаил
with open('data/model/iris_model.pkl', 'wb') as file:
    pickle.dump((model,scaler), file)

# загружаем модель из файла
with open('data/model/iris_model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

# сопостовляем индекс с названием класса
class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}


# создаем  endpoint для классификации ирисов
@app.post("/predict/")
async def predict(item: IrisImput):
    input_data = [[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]]
    input_data_scaler = scaler.transform(input_data)
    prediction_index = model.predict(input_data_scaler)[0]
    prediction_class = class_names[prediction_index]
    return {"prediction": prediction_class}

# создаем endpoint возвращающий информационное сообщение
@app.get("/")
async def get ():
    return {
        "message": "For iris classification, send a POST request to the /predict endpoint.",
        "example_body": {
            "sepal_length": 1.6,
            "sepal_width": 4.4,
            "petal_length": 1.4,
            "petal_width": 3.6
        }
    }
