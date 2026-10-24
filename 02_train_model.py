
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

# 1. Load the dataset
file_path = r"D:\ADAS-EV-Action-Classification\data\ADAS_EV_Dataset.csv"
df = pd.read_csv(file_path)

# 2. Select the features and target
target = "ADAS_output"

features = [
    "speed_kmh",
    "acceleration_mps2",
    "brake_intensity",
    "battery_level",
    "energy_consumption",
    "regen_braking_usage",
    "lane_deviation",
    "obstacle_distance",
    "traffic_density",
    "weather_condition",
    "road_type",
    "steering_angle",
    "reaction_time"
]

X = df[features]
y = df[target]

# 3. Split chronologically: first 70% for training,
#    final 30% for testing
split_index = int(len(df) * 0.70)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Identify categorical columns
categorical_features = ["weather_condition", "road_type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categories",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# 5. Train a simple baseline model
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("--- Baseline Model ---")
print("Accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 4
))

# 6. Train a Random Forest model
model = make_pipeline(
    preprocessor,
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
)

model.fit(X_train, y_train)

# 7. Predict actions for the test data
predictions = model.predict(X_test)

# 8. Evaluate the model
print("\n--- Random Forest Results ---")
print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))


# 9. Investigate model predictions
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

# Display incorrectly classified records
errors = results[results["Actual"] != results["Predicted"]]

print("\n--- Prediction Error Analysis ---")
print("Total test records:", len(results))
print("Incorrect predictions:", len(errors))
print("Correct predictions:", len(results) - len(errors))

print("\nIncorrectly Classified Records:")
print(errors)


# 10. Analyse feature importance

rf = model.named_steps["randomforestclassifier"]

feature_names = (
    model.named_steps["columntransformer"]
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- Top 10 Important Features ---")
print(importance.head(10).round(4).to_string(index=False))