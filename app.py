from flask import Flask,request,render_template
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open("loan_approval_model (1).pkl","rb"))


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=["POST"])

@app.route('/predict', methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        features = [
            data['no_of_dependents'],
            data['education'],
            data['self_employed'],
            data['income_annum'],
            data['loan_amount'],
            data['loan_term'],
            data['cibil_score'],
            data['residential_assets_value'],
            data['commercial_assets_value'],
            data['luxury_assets_value'],
            data['bank_asset_value']
        ]

        input_data = np.array([features])

        prediction = model.predict(input_data)[0]

        result = "Approval" if prediction == 1 else "Rejected"

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}
    
if __name__=="__main__":
    app.run(debug=True)


    