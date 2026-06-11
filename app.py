import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
app = Flask(__name__)

# Load model
model = pickle.load(open("loan_approval_model (1).pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')


# ---------------- API (Postman / Render testing) ----------------
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']

        features = [
            int(data['no_of_dependents']),
            int(data['education']),
            int(data['self_employed']),
            int(data['income_annum']),
            int(data['loan_amount']),
            int(data['loan_term']),
            int(data['cibil_score']),
            int(data['residential_assets_value']),
            int(data['commercial_assets_value']),
            int(data['luxury_assets_value']),
            int(data['bank_asset_value'])
        ]

        input_data = np.array(features).reshape(1, -1)

        prediction = model.predict(input_data)[0]

        result = "Approved" if prediction == 1 else "Rejected"

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------- Web Form (HTML UI) ----------------
@app.route('/predict', methods=['POST'])
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

        input_data = np.array(features).reshape(1, -1)

        prediction = model.predict(input_data)[0]

        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

        return render_template("home.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)