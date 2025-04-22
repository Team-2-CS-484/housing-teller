from flask import Flask, render_template, request
import joblib
import numpy as np
import requests
import pandas as pd

app = Flask(__name__)
model = joblib.load("xgb_boston_model.pkl")

chic_model = joblib.load("lgbm_chicago_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            features = []
            for field in ["lstat", "rm", "ptratio", "indus", "tax", "nox", "crim"]:
                val = request.form.get(field)
                if val == "" or val is None:
                    raise ValueError(f"{field} is missing.")
                features.append(float(val))

            pred = model.predict([features])[0]
            prediction = round(pred, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template("main_page.html", prediction=prediction)

@app.route("/chicago", methods=["GET", "POST"])
def chicago():
    prediction = None

    if request.method == "POST":
        try:
            lat, lon = request.form.get('lat'), request.form.get('lon')

            print(f"{lat},{lon}")

            url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"

            params = {
                'benchmark': 4,
                'vintage':4,
                'layers': 8,
                'format': 'json'}
            
            response = requests.get(url, params={**params, 'x': lat, 'y': lon}).json()
            geoid = response['result']['geographies']['Census Tracts'][0]['GEOID']

            ref_table = pd.read_csv("chicago_ref_table.csv")
            
            cols = ['EKW_2024', 'INC_2019-2023', 'CZM_2023', 'EDB_2019-2023']

            row = (
                ref_table
                .loc[ref_table['GEOID'].astype(int) == int(geoid), cols]
                .iloc[0]
            )

            usr_input = row.to_dict()

            for res_t in ['residence_type_1.0', 'residence_type_2.0', 'residence_type_3.0', 'residence_type_>=4.0']:
                usr_input[res_t] = False

            residence_type = int(request.form.get('Residence Type'))

            match residence_type:
                case 1:
                    usr_input['residence_type_1.0'] = True
                case 2:
                    usr_input['residence_type_2.0'] = True
                case 3:
                    usr_input['residence_type_3.0'] = True
                case _:
                    usr_input['residence_type_>=4.0'] = True

            features = ['Land Square Feet','Building Square Feet','Rooms','Bedrooms','Full Baths','Age','Total Garage Size']

            for f in features:
                usr_input[f] = float(request.form.get(f))
            
            usr_input['Building/Land Ratio'] = usr_input['Building Square Feet'] / usr_input['Land Square Feet']

            col_order = [
                'Land Square Feet',
                'Building Square Feet',
                'Rooms',
                'Bedrooms',
                'Full Baths',
                'Age',
                'EKW_2024',
                'INC_2019-2023',
                'CZM_2023',
                'EDB_2019-2023',
                'residence_type_1.0',
                'residence_type_2.0',
                'residence_type_3.0',
                'Total Garage Size',
                'Building/Land Ratio',
                'residence_type_>=4.0',]
            
            final_feature = [usr_input[x] for x in col_order]

            print(", ".join((map(str, final_feature))))

            prediction = round(chic_model.predict([final_feature])[0], 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("chicago_page.html", prediction=prediction)
    

if __name__ == "__main__":
    app.run(debug=True)
