import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from catboost import CatBoostClassifier, Pool

data = pd.read_csv("Studentdata.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
test_pool = Pool(X_test, y_test, cat_features=categorical_cols)

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
        "random_strength": trial.suggest_float("random_strength", 1, 20),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 0,
        "random_seed": 42
    }
    model = CatBoostClassifier(cat_features=categorical_cols, **params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
    return scores.mean()

print("🔎 Running Optuna hyperparameter tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("✅ Best Hyperparameters:", study.best_params)

best_params = study.best_params
best_params.update({
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": 200
})

model = CatBoostClassifier(cat_features=categorical_cols, **best_params)
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n📊 Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv_score = cross_val_score(model, X, y, cv=5, scoring="f1")
print("Cross-validation F1 mean:", cv_score.mean())

print("\n🔑 Feature Importances:")
feature_importances = model.get_feature_importance(prettified=True)
print(feature_importances)

model.save_model("best_catboost_model.cbm")
import joblib
joblib.dump(model, "best_catboost_model.pkl")

print("✅ Model saved successfully!")