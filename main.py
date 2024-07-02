from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np
app=Flask(__name__)

data=pd.read_csv('Cleanded_data.csv')

pipe=pickle.load(open("pipe.pkl",'rb'))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    print(locations)
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    locations= request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    total_sqft=request.form.get('total_sqft')
    print(locations,bath,bhk,total_sqft)
    input=pd.DataFrame([[locations,total_sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction=pipe.predict(input)[0]
    print(prediction)
    return str(np.round(prediction*1e5,2))


if __name__=="__main__":
    app.run(debug=True,port=5002)