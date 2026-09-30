"""Predict the direct tensile strength (DTS) of UHPFRC with the four optimised XGBoost models.

Usage:  python predict.py
Edit the MIX dictionary below. Units: kg/m3 except AR (-) and FTS (MPa).
Requires: pip install xgboost numpy
"""
import json, os
import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
MIX = {"Cement": 663, "Min. Adm.": 409, "Sand": 955, "SF": 156, "PEF": 0,
       "AR": 65, "FTS": 1900, "Water": 214.4, "SP": 10.72}

def features(m):
    f = dict(m)
    f["Binder"] = m["Cement"] + m["Min. Adm."]
    f["W/B"] = m["Water"] / f["Binder"]
    f["SP/B"] = m["SP"] / f["Binder"]
    f["Powder"] = f["Binder"] + m["Sand"]
    f["W/P"] = m["Water"] / f["Powder"]
    f["Vf (%)"] = (m["SF"] / 7850 + m["PEF"] / 970) * 100
    f["FRI"] = f["Vf (%)"] * m["AR"]
    return f

sc = json.load(open(os.path.join(HERE, "models", "scaler.json")))
f = features(MIX)
x = np.array([[(f[k] - sc["data_min"][k]) / (sc["data_max"][k] - sc["data_min"][k]) for k in sc["feature_order"]]])
out = [k for k in sc["feature_order"] if not sc["data_min"][k] <= f[k] <= sc["data_max"][k]]
for name in ["WOA", "GTO", "PSO", "GWO"]:
    b = xgb.Booster(); b.load_model(os.path.join(HERE, "models", f"{name}-XGB.json"))
    print(f"{name}-XGB : {float(b.predict(xgb.DMatrix(x))[0]):.3f} MPa")
if out: print("Warning - outside the training range:", ", ".join(out))
