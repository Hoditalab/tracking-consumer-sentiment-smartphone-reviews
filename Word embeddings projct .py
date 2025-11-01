#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from collections import Counter
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from datetime import datetime
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report
from adjustText import adjust_text
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')


# In[2]:


#  random seed for reproducibility
np.random.seed(42)

# Load dataset with error handling
file_path = r"C:\\Users\\dell\\Documents\\embeddings meaning lessons\\PROJECT 2\\processed_dataset_4_Hoda1.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    exit()

# Drop rows with missing review text
df.dropna(subset=['token_processed_body'], inplace=True)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')

# Keep relevant columns
df = df[['token_processed_body', 'date', 'rating']]

custom_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
}

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Tokenize, clean, and lemmatize text."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords and word.isalnum()]
    return tokens

# Apply preprocessing
df['tokens'] = df['token_processed_body'].apply(preprocess_text)


# In[3]:


# Word2Vec model
word2vec_model = Word2Vec(
    df['tokens'], 
    vector_size=200,
    window=5, 
    min_count=10,
    workers=4
)
word2vec_model.train(df['tokens'], total_examples=len(df['tokens']), epochs=20)

# Define key features
key_features = ['battery', 'camera', 'performance', 'software', 'durability', 'price']

# Extract similar terms
feature_terms = {}
for feature in key_features:
    if feature in word2vec_model.wv:
        similar_words = [word for word, similarity in word2vec_model.wv.most_similar(feature, topn=10)]
        feature_terms[feature] = similar_words

print("\n🔹 Key Features Identified:\n", feature_terms)


# In[19]:


# Validate Word2Vec: Similarity Scores
word_pairs = [('battery', 'charge'), ('camera', 'selfie'), ('price', 'budget')]
for w1, w2 in word_pairs:
    if w1 in word2vec_model.wv and w2 in word2vec_model.wv:
        print(f"Similarity ({w1} - {w2}): {word2vec_model.wv.similarity(w1, w2):.4f}")


# In[21]:


def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

df['sentiment'] = df['token_processed_body'].apply(get_sentiment)
sentiment_counts = df['sentiment'].value_counts()

print("\nSentiment Distribution:")
print(sentiment_counts)

# Generate pie chart (labels outside)
plt.figure(figsize=(5, 5))  # Smaller chart
plt.pie(sentiment_counts.values, 
        autopct='%1.1f%%',  # Show percentages inside the slices
        colors=['#2ecc71', '#e74c3c', '#3498db'],  # Green for Positive, Red for Negative, Blue for Neutral
        startangle=90,  # Start at the top
        textprops={'fontsize': 10})  # Readable font size for percentages
plt.title("Sentiment Distribution", fontsize=14, pad=15)
plt.axis('equal')  # Equal aspect ratio for circular pie

# Place labels in the legend outside the chart
legend_labels = sentiment_counts.index.tolist()  # Sentiment categories
plt.legend(legend_labels, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

# Save the plot as an image (optional, uncomment to use)
# plt.savefig('sentiment_distribution_pie.png', dpi=300, bbox_inches='tight')

plt.show()


# In[4]:


def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

df['sentiment'] = df['token_processed_body'].apply(get_sentiment)
sentiment_counts = df['sentiment'].value_counts()

print("\nSentiment Distribution:")
print(sentiment_counts)


# In[5]:


# Sentiment trends over time
df['year'] = df['date'].dt.year
yearly_sentiment = df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
yearly_sentiment_percent = yearly_sentiment.div(yearly_sentiment.sum(axis=1), axis=0) * 100

plt.figure(figsize=(12, 6))
for sentiment in yearly_sentiment.columns:
    plt.plot(yearly_sentiment_percent.index, yearly_sentiment_percent[sentiment], marker='o', label=sentiment)
plt.title("Sentiment Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Percentage of Sentiment")
plt.legend()
plt.grid()
plt.show()


# In[6]:


# Aspect-based sentiment analysis
def get_feature_sentiment(review_text, tokens):
    feature_sentiments = {feature: [] for feature in key_features}
    sentences = sent_tokenize(review_text)
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for feature in key_features:
            if feature in sentence_lower:
                polarity = TextBlob(sentence).sentiment.polarity
                sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
                feature_sentiments[feature].append(sentiment)
    return feature_sentiments

df['feature_sentiments'] = df.apply(lambda row: get_feature_sentiment(row['token_processed_body'], row['tokens']), axis=1)

feature_sentiments_agg = {feature: {"Positive": 0, "Negative": 0, "Neutral": 0} for feature in key_features}
for fs in df['feature_sentiments']:
    for feature, sentiments in fs.items():
        for sentiment in sentiments:
            feature_sentiments_agg[feature][sentiment] += 1

feature_sentiments_df = pd.DataFrame(feature_sentiments_agg).T
feature_sentiments_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
plt.title("Sentiment Distribution by Feature (Aspect-Based)")
plt.ylabel("Number of Mentions")
plt.xlabel("Feature")
plt.legend(title="Sentiment")
plt.xticks(rotation=45)
plt.show()


# In[7]:


def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    else:
        return "Neutral"

df['true_sentiment'] = df['rating'].apply(rating_to_sentiment)
true_labels = df['true_sentiment']
predicted_labels = df['sentiment']
accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels)

print(f"\n🔹 Sentiment Analysis Validation:")
print(f"Accuracy: {accuracy:.2f}")
print("Detailed Report:")
print(report)


# In[10]:


from adjustText import adjust_text
import matplotlib.pyplot as plt


words = list(word2vec_model.wv.index_to_key)[:30]
word_vectors = np.array([word2vec_model.wv[word] for word in words])


pca = PCA(n_components=2)
word_vectors_2D = pca.fit_transform(word_vectors)

# Categorize words for color coding
key_features_set = set(key_features) | {'screen', 'charge', 'iphone', 'android', 'sim', 'sound','durability', 'software',
                                       'battrey'}  
sentiment_words = {
    'good', 'great', 'love', 'like', 'well', 'quality', 'fast', 'problem', 'try', 'bad', 'poor', 'issue',
    'awesome', 'excellent', 'terrible', 'horrible'
}
colors = []
for word in words:
    if word in key_features_set:
        colors.append('blue')
    elif word in sentiment_words:
        colors.append('green')
    else:
        colors.append('gray')

# Plot using Matplotlib with increased figure size
plt.figure(figsize=(14, 8))
scatter = plt.scatter(word_vectors_2D[:, 0], word_vectors_2D[:, 1], c=colors, marker='o', alpha=0.7, s=100)

# Add labels with automatic adjustment and refined arrow properties
texts = [plt.text(word_vectors_2D[i, 0], word_vectors_2D[i, 1], word, fontsize=12) for i, word in enumerate(words)]
adjust_text(texts, 
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.3, shrinkA=15, shrinkB=15, alpha=0.3),
            force_points=0.5,  
            force_text=0.5)   

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Key Feature', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Sentiment', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='gray', markersize=10),
]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)

# Set title and subtitle with increased font size and padding
plt.suptitle("Word Embeddings Visualization (PCA)", fontsize=18, y=1.02)  
plt.title("PCA 1 and PCA 2 represent the two principal components", fontsize=14, pad=20)  

plt.xlabel("PCA 1", fontsize=12)
plt.ylabel("PCA 2", fontsize=12)
plt.grid(True, color='lightgray', linestyle='--', alpha=0.3)
plt.xlim(-14, 14)  
plt.ylim(-10, 10)  
plt.tight_layout()

plt.show()


# In[ ]:




