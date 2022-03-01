from flask import Flask, render_template,request
import numpy as np
import pickle
import xgboost
import sklearn

app=Flask(__name__)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route("/confirm", methods=['POST','GET'])
def predict():
    if request.method=='POST':
        income=float(request.form.get('income'))
        income=np.power(income,-0.20280658936585644,dtype=float)
        Income_log=(income-0.4901766534406379)/(0.06735802053503631)
        limit=float(request.form.get('limit'))
        limit=np.power(limit,0.39614936118102145,dtype=float)
        Limit_log=(limit-27.74271074744698)/(5.585085279010829)
        rating=float(request.form.get('rating'))
        rating=np.power(rating,0.26853943981127637,dtype=float)
        Rating_log=(rating-4.749713648307949)/(0.5672648658390562)
        Student_Yes=request.form.get('student')
        if Student_Yes=='Yes':
            Student_Yes=1
        else:
            Student_Yes=0
        Cards=float(request.form.get('cards'))
        Cards=(Cards-2.9575)/(1.371274858240354)
        
        data=[[Income_log,Limit_log,Rating_log,Student_Yes,Cards]]
        model_credit=pickle.load(open('model_credit.sav','rb'))  
        pred=model_credit.predict(data)

        return render_template('final.html',output=pred)

if __name__=="__main__":
    app.run(debug=True)
