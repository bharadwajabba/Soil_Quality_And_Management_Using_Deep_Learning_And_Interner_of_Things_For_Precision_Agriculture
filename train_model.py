import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib


data = pd.read_csv('dataset1.csv')
X = data.drop('Output', axis=1)
y = data['Output']


X = X.apply(lambda x: np.log1p(x) if np.issubdtype(x.dtype, np.number) else x)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


base_models = [
    ('hgb', HistGradientBoostingClassifier(random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500, random_state=42))
]

model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
model.fit(X_train, y_train)


joblib.dump(model, 'model/model.pkl')
print("Model trained and saved successfully!")