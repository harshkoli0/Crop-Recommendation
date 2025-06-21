import pandas as pd
df=pd.read_csv(r"C:\Users\hkoli\Downloads\Crop_recommendation.csv")
df.head()

df.columns=df.columns.str.strip()
df.isnull().sum()

df["label"].value_counts()

import seaborn as sns
heat=df.drop("label",axis=1)
sns.heatmap(heat.corr(),annot=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

x=df.drop("label",axis=1)
y=df["label"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=45)
models=RandomForestClassifier()
models.fit(x_train,y_train)
y_pred=models.predict(x_test)
accuracy_score(y_pred,y_test)
print(classification_report(y_pred,y_test))

feature=pd.DataFrame(models.feature_importances_,index=x.columns)
feature

from sklearn import tree
import matplotlib.pyplot as plt

tree.plot_tree(models.estimators_[0],feature_names=x_train.columns,filled=True)
plt.figure(figsize=(40, 20)) 
plt.show()

import pickle
import streamlit as st
import numpy as np


with open("crop.pkl", "wb") as f:
    pickle.dump(models, f)
    
st.title("🌾 Crop Recommendation App")

Nitrogen=st.slider("Nitrogen",0,100,30)
Phosphorus=st.slider("Phosphorus",0,100,30)
Potassium=st.slider("Potassium",0,100,30)
temperature=st.number_input("temperature (c)",0.0, 45.0, 30.0)
humidity=st.number_input("humidity %",10.0, 100.0, 60.0)
ph=st.number_input("soli ph",3.5, 9.5, 6.5)
rainfall=st.number_input("rainfall (mm)",20.0, 300.0, 100.0)

model = pickle.load(open("crop.pkl", "rb"))
data=np.array([[Nitrogen,Phosphorus,Potassium,temperature,humidity,ph,rainfall]])
pred=model.predict(data)


if st.button("Recommend Crop"):
    st.success(f"Recommend Crop: {pred[0]}")
else:
    print("thanks") 
st.sidebar.subheader("Acuracy score")  
st.sidebar.markdown(f"Acuracy_score: {accuracy_score(y_test,y_pred):.2f}")



if st.sidebar.button("Show Crop Labels"):
    st.sidebar.write(df["label"].unique())



