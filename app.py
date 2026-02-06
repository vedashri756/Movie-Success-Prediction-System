from flask import Flask, render_template, request
import pickle
import numpy as np

with open("genre_classes.pkl", "rb") as f:
    genre_classes = pickle.load(f)

app = Flask(__name__)

# Load trained model
with open("movie_success_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    budget = float(request.form["budget"])
    runtime = float(request.form["runtime"])
    vote_average = float(request.form["vote_average"])
    vote_count = float(request.form["vote_count"])

    # Basic validation
    if budget <= 0 or runtime <= 0 or vote_count < 0:
        return render_template(
            "index.html",
            prediction_text="Invalid input values"
        )

    # 🔹 NEW PART: get selected genres from form
    selected_genres = request.form.getlist("genres")

    # 🔹 NEW PART: build genre vector in correct order
    genre_vector = [1 if g in selected_genres else 0 for g in genre_classes]

    # 🔹 NEW PART: combine numeric + genre features
    features = np.array(
        [[budget, runtime, vote_average, vote_count] + genre_vector]
    )

    # Probability-based prediction
    prob_success = model.predict_proba(features)[0][1]

    threshold = 0.6
    prediction = 1 if prob_success >= threshold else 0

    result = (
        "Predicted Outcome: Box Office Success 🎬"
        if prediction == 1
        else "Predicted Outcome: Not a Box Office Success ❌"
    )

    confidence = f"Model Confidence: {prob_success*100:.2f}%"

    # Feature importance (Random Forest)
    rf_model = model.named_steps["rf"]
    importances = rf_model.feature_importances_

    feature_names = (
        ["Budget", "Runtime", "Average Rating", "Vote Count"]
        + list(genre_classes)
    )

    importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:6]   # show top 6 only

    return render_template(
        "index.html",
        prediction_text=result,
        confidence_text=confidence,
        importance=importance_pairs
    )


if __name__ == "__main__":
    app.run(debug=True)
