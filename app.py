from flask import Flask,request,render_template
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open("loan_approval_model (1).pkl","rb"))


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=["POST"])
def predict():
    try:
        features = [
    int(request.form['no_of_dependents']),
    int(request.form['education']),
    int(request.form['self_employed']),
    int(request.form['income_annum']),
    int(request.form['loan_amount']),
    int(request.form['loan_term']),
    int(request.form['cibil_score']),
    int(request.form['residential_assets_value']),
    int(request.form['commercial_assets_value']),
    int(request.form['luxury_assets_value']),
    int(request.form['bank_asset_value'])
    ]
        input=np.array([features])
        prediction=model.predict(input)[0]
        if prediction==1:
            result="Approval"
        else:
            result="Rejected"
        return render_template("home.html",prediction_text=result)
    except Exception as e:
        return str(e)
    
if __name__=="__main__":
    app.run(debug=True)


    