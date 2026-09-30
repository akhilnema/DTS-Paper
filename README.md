# UHPFRC direct tensile strength – optimised XGBoost models

Trained models and a graphical predictor accompanying:

**Nema, A., A. Tabani, and R. Biswas. "Direct Tensile Strength of Ultra-High-Performance Fiber-Reinforced Concrete: Interpretable Prediction Using Metaheuristic-Optimized XGBoost."** (manuscript under review).

The repository provides the four metaheuristic-optimised XGBoost models reported in the paper (GTO, WOA, PSO and GWO; WOA-XGB is the proposed model) so that predictions can be reproduced for any mix. The training dataset and the training code are available from the corresponding author on reasonable request.

## Contents

| Path | Description |
|---|---|
| `gui/UHPFRC_DTS_Predictor.html` | Stand-alone predictor. Open in any web browser; works offline, no installation. |
| `predict.py` | Command-line prediction with all four models. |
| `models/<name>-XGB.json` | Models in native XGBoost format (recommended; load with any XGBoost version). |
| `models/<name>-XGB.pkl`, `models/MinMaxScaler.pkl` | Same models and the scaler as scikit-learn pickles (XGBoost 3.2.0). |
| `models/scaler.json` | Min-max scaling limits for each of the 16 features. |
| `models/model_card.json` | Feature order, units, derived-feature formulas, hyperparameters and test metrics. |

## Inputs

Nine mix constituents are entered: cement, mineral admixture, sand, water and superplasticiser (kg/m³); steel fibre and PE fibre (kg/m³); fibre aspect ratio (–); fibre tensile strength (MPa). Seven features are derived automatically: binder, W/B, SP/B, powder, W/P, fibre volume Vf (%) and fibre reinforcement index FRI = Vf × AR. Inputs outside the range of the 296 training mixes trigger a warning.

## Quick start

```
pip install xgboost numpy
python predict.py
```

Predictions from the GUI, `predict.py` and the pickled models agree to 0.001 MPa.

## Licence

MIT. Please cite the paper above if you use these models.
