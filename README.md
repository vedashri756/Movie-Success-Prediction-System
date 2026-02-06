# Movie-Success-Prediction-System
An end-to-end data science and machine learning web application that predicts whether a movie is likely to be a box-office success based on budget, runtime, audience engagement metrics, and genres.
The system provides probability-based predictions, confidence scores, and feature-level explanations, and is deployed as a Flask web application.

📌 Problem Statement

Movie production involves significant financial risk.
This project aims to predict whether a movie will be commercially successful using pre-release and early-release attributes, enabling data-driven decision-making.

A movie is considered successful if its revenue is at least 2× its production budget, making this a business-driven binary classification problem.

📊 Dataset

Source: TMDB Movies Metadata Dataset

Attributes used:

Budget

Revenue

Runtime

Average Rating

Vote Count

Genres (multi-label)

The dataset contains real-world inconsistencies such as missing values and mixed data types, which were handled during preprocessing.

⚙️ Methodology
1. Data Preprocessing

Converted inconsistent data types

Removed invalid and missing values

Defined a profit-based success metric

Handled class imbalance using stratified sampling and class weighting

2. Feature Engineering

Numerical features: budget, runtime, ratings, vote count

Categorical feature: genres encoded using multi-label binarization

Combined numerical and genre features into a unified feature space

3. Model Building

Model: Random Forest Classifier

Pipeline includes:

Feature scaling (StandardScaler)

Class imbalance handling

Probability-based prediction

Decision threshold tuned for realistic business behavior

4. Model Evaluation

Evaluated using precision, recall, and F1-score

Reduced bias caused by class imbalance

Extracted feature importance for explainability

🌐 Web Application

The trained model is deployed using Flask and provides:

User input form for movie attributes

Probability-based prediction of success

Model confidence score

Top influencing features affecting the prediction

This converts the machine learning model into a usable decision-support system.

🛠️ Tech Stack

Programming Language: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn

Web Framework: Flask

📁 Project Structure
movie-success-prediction/
│
├── app.py
├── movie_success_prediction.ipynb
├── movie_success_model.pkl
├── genre_classes.pkl
├── requirements.txt
├── README.md
└── templates/
    └── index.html

▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/vedashri756/movie-success-prediction.git
cd movie-success-prediction

2. Install dependencies
pip install -r requirements.txt

3. Run the web application
python app.py

4. Open in browser
http://127.0.0.1:5000

🔍 Key Learnings

Defining the correct business problem is more important than complex algorithms

Class imbalance can significantly bias predictions if ignored

Explainability improves trust in machine learning systems

Deploying models bridges the gap between analysis and real-world usage

🚀 Future Improvements

Profit margin prediction using regression

Advanced feature engineering using cast and crew metadata

Comparison with gradient boosting models

Cloud deployment for public access

👤 Author

Vedashri
MSc Data Science
VIT Chennai    
