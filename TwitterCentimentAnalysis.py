import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # Changed model type


from sklearn.metrics import f1_score, accuracy_score
import warnings

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('train.csv')


# Display first few rows and datatype info
print(df.head())
print(df.info())



#####################################################

# Preprocessing functions
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

def preprocess_text(text):
    text = re.sub(r'@[\w]*', '', text)  # Remove Twitter handles
    text = re.sub(r'[^a-zA-Z#]', ' ', text)  # Remove special characters, numbers, and punctuations
    text = ' '.join([w for w in text.split() if len(w) > 3])  # Remove short words
    return text

# Apply preprocessing
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
df['clean_tweet'] = df['clean_tweet'].apply(preprocess_text)

# Tokenization and stemming
stemmer = PorterStemmer()
df['tokenized_tweet'] = df['clean_tweet'].apply(lambda x: nltk.word_tokenize(x.lower()))  # Tokenize and convert to lowercase
df['stemmed_tweet'] = df['tokenized_tweet'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])  # Stemming

# Combine stemmed tokens back into a single sentence
df['clean_tweet'] = df['stemmed_tweet'].apply(lambda tokens: ' '.join(tokens))





#######################################################################



# Word cloud visualization
def plot_wordcloud(text_data, title):
    all_words = " ".join([tweet for tweet in text_data])
    wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Plot word clouds for all tweets, positive tweets, and negative tweets
plot_wordcloud(df['clean_tweet'], 'Word Cloud for All Tweets')
plot_wordcloud(df[df['label'] == 0]['clean_tweet'], 'Word Cloud for Positive Tweets')
plot_wordcloud(df[df['label'] == 1]['clean_tweet'], 'Word Cloud for Negative Tweets')




# Count hashtags
def count_hashtags(tweet):
    hashtags = re.findall(r'#(\w+)', tweet)
    return len(hashtags)

df['num_hashtags'] = df['tweet'].apply(count_hashtags)

# Plot histogram of hashtag counts
plt.figure(figsize=(10, 6))
sns.histplot(df['num_hashtags'], bins=range(6), kde=False)
plt.xlabel('Number of Hashtags')
plt.ylabel('Count')
plt.title('Histogram of Hashtag Counts')
plt.show()

# Count positive and negative words
positive_words = ['good', 'great', 'happy', 'awesome', 'love']
negative_words = ['bad', 'terrible', 'sad', 'awful', 'hate']

def count_positive_words(tweet):
    count = sum(tweet.lower().count(word) for word in positive_words)
    return count

def count_negative_words(tweet):
    count = sum(tweet.lower().count(word) for word in negative_words)
    return count

df['num_positive_words'] = df['clean_tweet'].apply(count_positive_words)
df['num_negative_words'] = df['clean_tweet'].apply(count_negative_words)

# Plot bar chart of positive and negative word counts
plt.figure(figsize=(10, 6))
sns.barplot(data=df[['num_positive_words', 'num_negative_words']], ci=None)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Positive and Negative Word Counts')
plt.xticks(ticks=[0, 1], labels=['Positive', 'Negative'])
plt.show()



# Count hashtags
def count_hashtags(tweet):
    hashtags = re.findall(r'#(\w+)', tweet)
    return hashtags

df['hashtags'] = df['tweet'].apply(count_hashtags)

# Flatten list of hashtags
all_hashtags = [hashtag for sublist in df['hashtags'] for hashtag in sublist]

# Count occurrences of each hashtag
hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)

# Plot top 10 hashtags
plt.figure(figsize=(10, 6))
sns.barplot(x=hashtag_counts.values, y=hashtag_counts.index, palette='viridis')
plt.xlabel('Count')
plt.ylabel('Hashtag')
plt.title('Top 10 Hashtags')
plt.show()

# Separate positive and negative tweets
positive_tweets = df[df['label'] == 0]
negative_tweets = df[df['label'] == 1]

# Flatten list of hashtags for positive and negative tweets
all_positive_hashtags = [hashtag for sublist in positive_tweets['hashtags'] for hashtag in sublist]
all_negative_hashtags = [hashtag for sublist in negative_tweets['hashtags'] for hashtag in sublist]

# Count occurrences of each positive hashtag
positive_hashtag_counts = pd.Series(all_positive_hashtags).value_counts().head(10)

# Count occurrences of each negative hashtag
negative_hashtag_counts = pd.Series(all_negative_hashtags).value_counts().head(10)

# Plot top positive hashtags
plt.figure(figsize=(10, 6))
sns.barplot(x=positive_hashtag_counts.values, y=positive_hashtag_counts.index, palette='coolwarm')
plt.xlabel('Count')
plt.ylabel('Hashtag')
plt.title('Top Positive Hashtags')
plt.show()

# Plot top negative hashtags
plt.figure(figsize=(10, 6))
sns.barplot(x=negative_hashtag_counts.values, y=negative_hashtag_counts.index, palette='coolwarm')
plt.xlabel('Count')
plt.ylabel('Hashtag')
plt.title('Top Negative Hashtags')
plt.show()


####################################################


# Feature extraction using CountVectorizer
vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['clean_tweet'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=1422)  # Increased number of estimators for better performance
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("F1 Score:", f1)
print("Accuracy:", accuracy)




###########################################################






# Load test.csv
test_df = pd.read_csv('test.csv')

# Add empty column named "label" at the second index position
test_df.insert(1, 'label', '')

# Preprocessing functions
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

def preprocess_text(text):
    text = re.sub(r'@[\w]*', '', text)  # Remove Twitter handles
    text = re.sub(r'[^a-zA-Z#]', ' ', text)  # Remove special characters, numbers, and punctuations
    text = ' '.join([w for w in text.split() if len(w) > 3])  # Remove short words
    return text

# Apply preprocessing
test_df['clean_tweet'] = np.vectorize(remove_pattern)(test_df['tweet'], "@[\w]*")
test_df['clean_tweet'] = test_df['clean_tweet'].apply(preprocess_text)

# Tokenization and stemming
stemmer = PorterStemmer()
test_df['tokenized_tweet'] = test_df['clean_tweet'].apply(lambda x: nltk.word_tokenize(x.lower()))  # Tokenize and convert to lowercase
test_df['stemmed_tweet'] = test_df['tokenized_tweet'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])  # Stemming

# Combine stemmed tokens back into a single sentence
test_df['clean_tweet'] = test_df['stemmed_tweet'].apply(lambda tokens: ' '.join(tokens))




# Feature extraction for test data using the same vectorizer
X_test = vectorizer.transform(test_df['clean_tweet'])

# Predict labels for the test data
test_df['label'] = model.predict(X_test)



# Save modified DataFrame to new test.csv file


test_df.to_csv('predictions.csv', index=False)

test_df.drop(['clean_tweet', 'tweet', 'tokenized_tweet', 'stemmed_tweet'], axis=1, inplace=True)

test_df.to_csv('test_predictions.csv', index=False)





############################################################

predicted_test=pd.read_csv('predictions.csv')









# Check for NaN values and replace them with an empty string
predicted_test['clean_tweet'] = predicted_test['clean_tweet'].replace(np.nan, '', regex=True)




# Count positive and negative words
positive_words = ['good', 'great', 'happy', 'awesome', 'love']
negative_words = ['bad', 'terrible', 'sad', 'awful', 'hate']

def count_positive_words(tweet):
    count = sum(tweet.lower().count(word) for word in positive_words)
    return count

def count_negative_words(tweet):
    count = sum(tweet.lower().count(word) for word in negative_words)
    return count

predicted_test['num_positive_words'] = predicted_test['clean_tweet'].apply(count_positive_words)
predicted_test['num_negative_words'] = predicted_test['clean_tweet'].apply(count_negative_words)

# Plot word clouds for positive and negative tweets
def plot_wordcloud(text_data, title):
    all_words = " ".join([tweet for tweet in text_data])
    wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

plot_wordcloud(predicted_test[predicted_test['label'] == 0]['clean_tweet'], 'Word Cloud for Positive Tweets')
plot_wordcloud(predicted_test[predicted_test['label'] == 1]['clean_tweet'], 'Word Cloud for Negative Tweets')

# Count hashtags
def count_hashtags(tweet):
    hashtags = re.findall(r'#(\w+)', tweet)
    return hashtags

predicted_test['hashtags'] = predicted_test['tweet'].apply(count_hashtags)

# Flatten list of hashtags for positive and negative tweets
all_positive_hashtags = [hashtag for sublist in predicted_test[predicted_test['label'] == 0]['hashtags'] for hashtag in sublist]
all_negative_hashtags = [hashtag for sublist in predicted_test[predicted_test['label'] == 1]['hashtags'] for hashtag in sublist]

# Count occurrences of each positive hashtag
positive_hashtag_counts = pd.Series(all_positive_hashtags).value_counts().head(10)

# Count occurrences of each negative hashtag
negative_hashtag_counts = pd.Series(all_negative_hashtags).value_counts().head(10)

# Plot top positive hashtags
plt.figure(figsize=(10, 6))
positive_hashtag_counts.plot(kind='bar', color='blue')
plt.xlabel('Hashtag')
plt.ylabel('Count')
plt.title('Top Positive Hashtags')
plt.show()

# Plot top negative hashtags
plt.figure(figsize=(10, 6))
negative_hashtag_counts.plot(kind='bar', color='red')
plt.xlabel('Hashtag')
plt.ylabel('Count')
plt.title('Top Negative Hashtags')
plt.show()


