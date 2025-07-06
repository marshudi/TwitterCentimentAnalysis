# Twitter Sentiment Analysis

**Authors:**
1- Mohammed Amour Al-Marshudi  <br>
2- Israa Dawood Al-Rashdi  <br>
3- Rawanq Mohammed Al-Harthy  <br>
**Date:** 05-May-2024

## Project Overview

This repository implements a machine learning pipeline for sentiment analysis on Twitter data. It includes data loading, preprocessing (cleaning, tokenization, stemming), feature extraction, model training, evaluation, prediction generation, and visualization of sentiment trends. Detailed documentation is provided in both Word and PDF formats.

## Key Features

* **Data Loading:** Reads labeled training data (`train.csv`) and unlabeled test data (`test.csv`).
* **Text Preprocessing:** Removes Twitter handles, special characters, and tokens shorter than four characters; applies tokenization and Porter stemming.
* **Feature Extraction:** Converts text into numerical features using `CountVectorizer`.
* **Model Training:** Uses a `RandomForestClassifier` with configurable hyperparameters.
* **Evaluation:** Computes F1 score and accuracy to assess model performance.
* **Prediction Output:** Generates predictions saved to `predictions.csv` and `test_predictions.csv`.
* **Visualizations:** Produces word clouds, hashtag distributions, and sentiment word count charts.

## Repository Structure

```plaintext
├── train.csv                        # Labeled tweets for training (columns: id, label, tweet)
├── test.csv                         # Unlabeled tweets for prediction (columns: id, tweet)
├── TwitterCentimentAnalysis.py      # Main analysis and modeling script
├── predictions.csv                  # Predicted labels and cleaned text for training split
├── test_predictions.csv             # Final predictions for test.csv (id and label)
├── "Twitter Sentiment Analysis.docx"  # Detailed project report in Word format
├── "Twitter Sentiment Analysis.pdf"   # Detailed project report in PDF format
└── README.md                        # Project documentation (this file)
```

> **Note:** The file names for the report contain spaces and should be enclosed in quotes when accessed via command line.

## Requirements

* Python 3.7 or higher
* pandas
* numpy
* matplotlib
* seaborn
* nltk
* wordcloud
* scikit-learn

Install dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn
```

## Setup

1. **Download NLTK Resources**

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
2. **Place Data Files**
   Ensure `train.csv` and `test.csv` are in the project root.

## Usage

Run the analysis script:

```bash
python "TwitterCentimentAnalysis.py"
```

The script will:

1. Load and preprocess the data.
2. Train the Random Forest model.
3. Evaluate with F1 score and accuracy.
4. Save predictions to `predictions.csv` and `test_predictions.csv`.
5. Display visualizations of sentiment trends.

## Data Preprocessing Steps

1. **Remove Handles:** Strips `@username` patterns.
2. **Clean Text:** Keeps letters and hashtags, replaces other characters with spaces.
3. **Filter Tokens:** Removes words shorter than four characters.
4. **Tokenize:** Splits cleaned tweets into tokens.
5. **Stem:** Applies Porter stemming to each token.

## Model & Evaluation

* **Vectorizer:** `CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')`.
* **Classifier:** `RandomForestClassifier(n_estimators=100, random_state=1422)`.
* **Metrics:** F1 score and accuracy on held-out split.

## Visualizations

* **Word Clouds:** Top terms in positive vs. negative tweets.
* **Hashtag Analysis:** Frequency distribution by sentiment.
* **Sentiment Word Counts:** Counts of pre-defined positive/negative words.

## References

* Analytics Vidhya Twitter Sentiment Analysis Dataset: [https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/)
* Random Forest algorithm overview on Analytics Vidhya blog.


