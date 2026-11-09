import pickle
import pandas as pd

from fastapi import FastAPI

app = FastAPI()

with open('weather_forecast.bin', 'rb') as f_in:
    artifacts = pickle.load(f_in)


def predict_rain(record, artifacts):
    df_record = pd.DataFrame([record])
    df_record = df_record.reindex(columns=list(artifacts['train_medians'].index) + artifacts['categorical'])
    df_record = df_record.fillna(artifacts['train_medians'])
    df_record[artifacts['categorical']] = df_record[artifacts['categorical']].fillna('Missing')

    X = artifacts['dv'].transform(df_record.to_dict(orient='records'))
    probability = float(artifacts['model'].predict_proba(X)[0, 1])

    return probability

@app.post("/predict")
def predict(record: dict):
    probability = predict_rain(record, artifacts)
    return {
        "rain_probability": probability,
        "rain_tomorrow": "yes" if probability >= 0.5 else "no",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9696)