from sklearn.preprocessing import MinMaxScaler
import joblib

def scale_features(df, feature_cols, scaler_path="models/saved_models/feature_scaler.pkl", fit=False):
    scaler = None
    if fit:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df
