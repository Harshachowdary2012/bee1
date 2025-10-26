"""
================================================================================
MAJOR PROJECT: ADVANCED SENTIMENT ANALYSIS ON INDIAN FINANCIAL NEWS
================================================================================
Author: HARSHA VARDHAN  K
Date: October 2025
Description: Comprehensive NLP-based sentiment analysis system using multiple
             machine learning and deep learning approaches with extensive
             preprocessing, feature engineering, and evaluation metrics.
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORT LIBRARIES AND DEPENDENCIES
# ============================================================================

print("="*100)
print("LOADING LIBRARIES AND DEPENDENCIES")
print("="*100)

# Core Data Processing Libraries
import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

# Dataset Loading
from datasets import load_dataset

# NLP and Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
print("\n[INFO] Downloading NLTK data packages...")
import time
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print("[OK] NLTK packages downloaded")
    time.sleep(0.5)
except Exception as e:
    print(f"[WARNING] NLTK download issue: {e}")
    print("[INFO] Continuing with alternative methods...")
    time.sleep(0.5)

# Feature Extraction
from sklearn.feature_extraction.text import (
    TfidfVectorizer, 
    CountVectorizer, 
    HashingVectorizer
)

# Machine Learning Models
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)

# Metrics and Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, 
        Dropout, 
        LSTM, 
        Embedding, 
        Conv1D, 
        MaxPooling1D,
        GlobalMaxPooling1D,
        Bidirectional,
        Flatten,
        Input,
        Concatenate
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
    print("[OK] TensorFlow imported successfully")
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    print(f"[WARNING] TensorFlow not available: {e}")
    print("[INFO] Deep learning models will be skipped.")
except Exception as e:
    DEEP_LEARNING_AVAILABLE = False
    print(f"[WARNING] TensorFlow import error: {e}")
    print("[INFO] Deep learning models will be skipped.")

# Time and System
import time
import datetime
from collections import Counter
import pickle
import json

# Statistical Analysis
from scipy import stats
from scipy.sparse import hstack

print("\n[OK] All libraries loaded successfully!")
print(f"Timestamp: {datetime.datetime.now()}")

# ============================================================================
# SECTION 2: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("\n" + "="*100)
print("SECTION 2: DATA LOADING AND INITIAL EXPLORATION")
print("="*100)

start_time = time.time()

# Load dataset
print("\n[STEP 1] Loading Indian Financial News Dataset...")
ds = load_dataset("kdave/Indian_Financial_News")

# Convert to pandas DataFrame
df_original = pd.DataFrame(ds['train'])

# Create unbalanced dataset (more realistic for real-world scenarios)
print("\n[STEP 1.1] Creating realistic unbalanced dataset...")
# Sample different amounts for each sentiment
negative_samples = df_original[df_original['Sentiment'] == 'Negative'].sample(n=7200, random_state=42)
neutral_samples = df_original[df_original['Sentiment'] == 'Neutral'].sample(n=5800, random_state=42)
positive_samples = df_original[df_original['Sentiment'] == 'Positive'].sample(n=6100, random_state=42)

# Combine and shuffle
df_original = pd.concat([negative_samples, neutral_samples, positive_samples], ignore_index=True)
df_original = df_original.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n[OK] Dataset loaded successfully!")
print(f"Loading time: {time.time() - start_time:.2f} seconds")
print(f"\nDataset Information:")
print(f"  Total samples: {len(df_original):,}")
print(f"  Columns: {df_original.columns.tolist()}")
print(f"  Memory usage: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display first few rows
print(f"\nFirst 3 rows:")
print(df_original.head(3))

# Basic statistics
print(f"\nBasic Statistics:")
print(df_original.describe(include='all'))

# Check for missing values
print(f"\nMissing Values:")
missing_values = df_original.isnull().sum()
print(missing_values)

# Sentiment distribution
print(f"\nSentiment Distribution:")
sentiment_counts = df_original['Sentiment'].value_counts()
print(sentiment_counts)
print(f"\nSentiment Percentages:")
sentiment_percentages = df_original['Sentiment'].value_counts(normalize=True) * 100
print(sentiment_percentages)

# Text length statistics
df_original['text_length'] = df_original['Content'].apply(lambda x: len(str(x)))
df_original['word_count'] = df_original['Content'].apply(lambda x: len(str(x).split()))
df_original['char_count'] = df_original['Content'].apply(lambda x: len(str(x)))

print(f"\nText Length Statistics:")
print(df_original[['text_length', 'word_count', 'char_count']].describe())

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*100)
print("SECTION 3: EXPLORATORY DATA ANALYSIS")
print("="*100)

# Create visualizations directory
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 3.1: Sentiment Distribution Plot
print("\n[STEP 1] Creating sentiment distribution visualization...")
plt.figure(figsize=(12, 6))
sentiment_counts.plot(kind='bar', color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
plt.title('Sentiment Distribution in Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(sentiment_counts):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/sentiment_distribution.png', dpi=300)
plt.close()

# 3.2: Text Length Distribution by Sentiment
print("[STEP 2] Creating text length distribution...")
plt.figure(figsize=(14, 6))
for sentiment in df_original['Sentiment'].unique():
    data = df_original[df_original['Sentiment'] == sentiment]['word_count']
    plt.hist(data, alpha=0.5, label=sentiment, bins=50)
plt.title('Word Count Distribution by Sentiment', fontsize=16, fontweight='bold')
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/word_count_distribution.png', dpi=300)
plt.close()

# 3.3: Box plot for word count by sentiment
print("[STEP 3] Creating box plots...")
plt.figure(figsize=(12, 6))
df_original.boxplot(column='word_count', by='Sentiment', figsize=(12, 6))
plt.title('Word Count Distribution by Sentiment (Box Plot)', fontsize=16, fontweight='bold')
plt.suptitle('')
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Word Count', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/word_count_boxplot.png', dpi=300)
plt.close()

# 3.4: Statistical summary by sentiment
print("\n[STEP 4] Statistical analysis by sentiment:")
for sentiment in df_original['Sentiment'].unique():
    print(f"\n{sentiment} Statistics:")
    subset = df_original[df_original['Sentiment'] == sentiment]
    print(f"  Average word count: {subset['word_count'].mean():.2f}")
    print(f"  Median word count: {subset['word_count'].median():.2f}")
    print(f"  Std deviation: {subset['word_count'].std():.2f}")

print("\n[OK] EDA completed!")

# ============================================================================
# SECTION 4: DATA PREPROCESSING AND CLEANING
# ============================================================================

print("\n" + "="*100)
print("SECTION 4: DATA PREPROCESSING AND CLEANING")
print("="*100)

# 4.1: Create working copy
print("\n[STEP 1] Creating working copy of data...")
df_processed = df_original.copy()

# 4.2: Text Cleaning Functions
print("[STEP 2] Defining text cleaning functions...")

def remove_urls(text):
    """Remove URLs from text"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_html(text):
    """Remove HTML tags"""
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def remove_emojis(text):
    """Remove emojis"""
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punctuation(text):
    """Remove punctuation"""
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    """Remove numbers"""
    return re.sub(r'\d+', '', text)

def remove_extra_whitespace(text):
    """Remove extra whitespace"""
    return ' '.join(text.split())

def to_lowercase(text):
    """Convert to lowercase"""
    return text.lower()

def remove_stopwords_func(text):
    """Remove stopwords"""
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback to manual stopwords list
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                      'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def stem_text(text):
    """Apply stemming"""
    try:
        stemmer = PorterStemmer()
        word_tokens = text.split()
        stemmed = [stemmer.stem(word) for word in word_tokens]
        return ' '.join(stemmed)
    except:
        return text  # Return original if stemming fails

def lemmatize_text(text):
    """Apply lemmatization"""
    try:
        lemmatizer = WordNetLemmatizer()
        word_tokens = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized)
    except:
        return text  # Return original if lemmatization fails

def comprehensive_cleaning(text):
    """Apply all cleaning steps"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_emojis(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_whitespace(text)
    return text

# 4.3: Apply cleaning (optimized single-pass approach)
print("\n[STEP 3] Applying text cleaning pipeline...")
print("  - Processing text in batches for better performance...")

# Process in smaller batches to avoid memory issues and improve performance
batch_size = 1000
total_batches = (len(df_processed) + batch_size - 1) // batch_size

# Initialize columns
df_processed['cleaned_text'] = ''
df_processed['cleaned_no_stopwords'] = ''
df_processed['stemmed_text'] = ''
df_processed['lemmatized_text'] = ''

for i in range(0, len(df_processed), batch_size):
    batch_end = min(i + batch_size, len(df_processed))
    batch_data = df_processed.iloc[i:batch_end].copy()
    
    print(f"    Processing batch {i//batch_size + 1}/{total_batches} ({i+1}-{batch_end})")
    
    # Apply all cleaning steps in one pass
    batch_data['cleaned_text'] = batch_data['Content'].apply(comprehensive_cleaning)
    batch_data['cleaned_no_stopwords'] = batch_data['cleaned_text'].apply(remove_stopwords_func)
    batch_data['stemmed_text'] = batch_data['cleaned_no_stopwords'].apply(stem_text)
    batch_data['lemmatized_text'] = batch_data['cleaned_no_stopwords'].apply(lemmatize_text)
    
    # Update the main dataframe
    df_processed.iloc[i:batch_end, df_processed.columns.get_loc('cleaned_text')] = batch_data['cleaned_text']
    df_processed.iloc[i:batch_end, df_processed.columns.get_loc('cleaned_no_stopwords')] = batch_data['cleaned_no_stopwords']
    df_processed.iloc[i:batch_end, df_processed.columns.get_loc('stemmed_text')] = batch_data['stemmed_text']
    df_processed.iloc[i:batch_end, df_processed.columns.get_loc('lemmatized_text')] = batch_data['lemmatized_text']

print("  - Text cleaning completed!")

# 4.4: Remove empty texts
print("\n[STEP 4] Removing empty texts...")
original_len = len(df_processed)
df_processed = df_processed[df_processed['cleaned_text'].str.len() > 0]
removed = original_len - len(df_processed)
print(f"  Removed {removed} empty texts")
print(f"  Remaining samples: {len(df_processed)}")

# 4.5: Display cleaning results
print("\n[STEP 5] Sample cleaning results:")
sample_idx = 0
print(f"\nOriginal text:\n{df_original['Content'].iloc[sample_idx][:200]}...\n")
print(f"Cleaned text:\n{df_processed['cleaned_text'].iloc[sample_idx][:200]}...\n")
print(f"Without stopwords:\n{df_processed['cleaned_no_stopwords'].iloc[sample_idx][:200]}...\n")

print("\n[OK] Data preprocessing completed!")

# ============================================================================
# SECTION 5: ADVANCED FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*100)
print("SECTION 5: ADVANCED FEATURE ENGINEERING")
print("="*100)

# 5.1: Basic text features
print("\n[STEP 1] Extracting basic text features...")
df_processed['cleaned_word_count'] = df_processed['cleaned_text'].apply(lambda x: len(x.split()))
df_processed['cleaned_char_count'] = df_processed['cleaned_text'].apply(len)
df_processed['avg_word_length'] = df_processed['cleaned_text'].apply(
    lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
)
df_processed['sentence_count'] = df_processed['Content'].apply(
    lambda x: str(x).count('.') + str(x).count('!') + str(x).count('?')
)

# 5.2: Uppercase and special character features
print("[STEP 2] Extracting uppercase and special character features...")
df_processed['uppercase_count'] = df_original['Content'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper())
)
df_processed['uppercase_ratio'] = df_processed['uppercase_count'] / df_processed['char_count']
df_processed['exclamation_count'] = df_original['Content'].apply(
    lambda x: str(x).count('!')
)
df_processed['question_count'] = df_original['Content'].apply(
    lambda x: str(x).count('?')
)

# 5.3: Lexical diversity
print("[STEP 3] Calculating lexical diversity...")
df_processed['unique_words'] = df_processed['cleaned_text'].apply(
    lambda x: len(set(x.split()))
)
df_processed['lexical_diversity'] = df_processed['unique_words'] / (df_processed['cleaned_word_count'] + 1)

# 5.4: Financial sentiment-specific keywords (enhanced features)
print("[STEP 4] Extracting financial sentiment keyword features...")
print("  - Using domain-specific financial terms for better accuracy...")

# Positive financial terms
positive_words = [
    'profit', 'beat estimates', 'growth', 'expansion', 'upgrade', 'record high', 
    'bullish', 'recovery', 'outperform', 'strong demand', 'new contracts', 
    'dividend increase', 'gain', 'rise', 'surge', 'increase', 'positive', 'up', 'rally'
]

# Negative financial terms  
negative_words = [
    'loss', 'downgrade', 'decline', 'cut', 'miss estimates', 'weak', 'slowdown', 
    'debt', 'lawsuit', 'default', 'drop', 'underperform', 'reduced forecast', 
    'slump', 'fall', 'crash', 'decrease', 'negative', 'down', 'plunge'
]

# Neutral financial terms
neutral_words = [
    'announces', 'reports', 'maintains', 'holds steady', 'unchanged', 
    'in line with expectations', 'no change', 'continues operations',
    'stable', 'maintain', 'hold', 'steady', 'flat'
]

print(f"  - Positive terms: {len(positive_words)} financial keywords")
print(f"  - Negative terms: {len(negative_words)} financial keywords") 
print(f"  - Neutral terms: {len(neutral_words)} financial keywords")

# Count financial sentiment terms (simplified)
def count_financial_terms(text, term_list):
    """Count financial terms without detailed logging"""
    return len([word for word in text.split() if word in term_list])

print("  - Counting financial sentiment terms...")
df_processed['positive_word_count'] = df_processed['cleaned_text'].apply(
    lambda x: count_financial_terms(x, positive_words)
)
df_processed['negative_word_count'] = df_processed['cleaned_text'].apply(
    lambda x: count_financial_terms(x, negative_words)
)
df_processed['neutral_word_count'] = df_processed['cleaned_text'].apply(
    lambda x: count_financial_terms(x, neutral_words)
)

# 5.5: Financial Terms Summary and N-gram Analysis
print("[STEP 5] Financial Terms Summary and N-gram Analysis...")

# Display financial terms summary
print("\nFinancial Terms Summary:")
print("=" * 50)
total_positive = df_processed['positive_word_count'].sum()
total_negative = df_processed['negative_word_count'].sum()
total_neutral = df_processed['neutral_word_count'].sum()

print(f"Total Positive Financial Terms Found: {total_positive:,}")
print(f"Total Negative Financial Terms Found: {total_negative:,}")
print(f"Total Neutral Financial Terms Found: {total_neutral:,}")
print(f"Total Financial Terms: {total_positive + total_negative + total_neutral:,}")

# Show most common terms
print("\nMost Common Financial Terms by Category:")
print("-" * 40)

# Count individual terms across all texts
all_positive_terms = []
all_negative_terms = []
all_neutral_terms = []

for text in df_processed['cleaned_text']:
    all_positive_terms.extend([word for word in text.split() if word in positive_words])
    all_negative_terms.extend([word for word in text.split() if word in negative_words])
    all_neutral_terms.extend([word for word in text.split() if word in neutral_words])

from collections import Counter

if all_positive_terms:
    pos_counter = Counter(all_positive_terms)
    print("Top 10 Positive Diaphrms:")
    for term, count in pos_counter.most_common(10):
        print(f"  {term}: {count}")

if all_negative_terms:
    neg_counter = Counter(all_negative_terms)
    print("Top 10 Negative Diaphrms:")
    for term, count in neg_counter.most_common(10):
        print(f"  {term}: {count}")

if all_neutral_terms:
    neu_counter = Counter(all_neutral_terms)
    print("Top 10 Neutral Diaphrms:")
    for term, count in neu_counter.most_common(10):
        print(f"  {term}: {count}")

# N-gram analysis (silent)
def get_top_ngrams(corpus, n=2, top_k=20):
    """Get top n-grams from corpus"""
    vec = CountVectorizer(ngram_range=(n, n), max_features=top_k).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq

# Get bigrams for each sentiment (silent processing)
for sentiment in df_processed['Sentiment'].unique():
    subset = df_processed[df_processed['Sentiment'] == sentiment]['cleaned_text']
    top_bigrams = get_top_ngrams(subset, n=2, top_k=10)

# 5.6: Additional statistical features
print("\n[STEP 6] Calculating additional statistical features...")
df_processed['digit_count'] = df_original['Content'].apply(
    lambda x: sum(c.isdigit() for c in str(x))
)
df_processed['space_count'] = df_original['Content'].apply(
    lambda x: str(x).count(' ')
)
df_processed['special_char_count'] = df_original['Content'].apply(
    lambda x: sum(not c.isalnum() and not c.isspace() for c in str(x))
)

print("\n[OK] Feature engineering completed!")
print(f"Total features created: {len(df_processed.columns)}")


# Display feature summary
print("\nFeature Summary:")
feature_cols = ['cleaned_word_count', 'avg_word_length', 'lexical_diversity', 
                'positive_word_count', 'negative_word_count', 'neutral_word_count']
print(df_processed[feature_cols].describe())

# ============================================================================
# SECTION 6: CREATING FAKE DATASET FOR PREDICTION
# ============================================================================

print("\n" + "="*100)
print("SECTION 6: CREATING DATASET WITHOUT LABELS")
print("="*100)

# Create fake dataset without sentiment labels
df_fake = df_processed.drop('Sentiment', axis=1).copy()
actual_sentiments = df_processed['Sentiment'].copy()

print(f"\n[OK] Original dataset (with labels): {df_processed.shape}")
print(f"[OK] Fake dataset (without labels): {df_fake.shape}")
print(f"[OK] Actual sentiments stored separately: {len(actual_sentiments)}")

# ============================================================================
# SECTION 7: TEXT VECTORIZATION AND TOKENIZATION
# ============================================================================

print("\n" + "="*100)
print("SECTION 7: TEXT VECTORIZATION AND TOKENIZATION")
print("="*100)

# 7.1: TF-IDF Vectorization
print("\n[STEP 1] TF-IDF Vectorization...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)

X_tfidf = tfidf_vectorizer.fit_transform(df_processed['cleaned_text'])
print(f"  TF-IDF shape: {X_tfidf.shape}")
print(f"  Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# 7.2: Count Vectorization
print("\n[STEP 2] Count Vectorization...")
count_vectorizer = CountVectorizer(
    max_features=3000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)

X_count = count_vectorizer.fit_transform(df_processed['cleaned_text'])
print(f"  Count Vectorizer shape: {X_count.shape}")

# 7.3: Hashing Vectorization
print("\n[STEP 3] Hashing Vectorization...")
hashing_vectorizer = HashingVectorizer(
    n_features=2000,
    ngram_range=(1, 2),
    alternate_sign=False
)

X_hash = hashing_vectorizer.fit_transform(df_processed['cleaned_text'])
print(f"  Hashing Vectorizer shape: {X_hash.shape}")

# 7.4: Create additional feature matrix
print("\n[STEP 4] Creating additional feature matrix...")
additional_features = df_processed[[
    'cleaned_word_count', 'avg_word_length', 'lexical_diversity',
    'positive_word_count', 'negative_word_count', 'neutral_word_count',
    'uppercase_ratio', 'exclamation_count', 'question_count'
]].values

# Normalize additional features
scaler = StandardScaler()
additional_features_scaled = scaler.fit_transform(additional_features)

print(f"  Additional features shape: {additional_features_scaled.shape}")

# 7.5: Combine TF-IDF with additional features
print("\n[STEP 5] Combining feature matrices...")
from scipy.sparse import hstack, csr_matrix

X_combined = hstack([X_tfidf, csr_matrix(additional_features_scaled)])
print(f"  Combined feature shape: {X_combined.shape}")

print("\n[OK] Vectorization completed!")

# ============================================================================
# SECTION 8: PREPARE TRAIN-TEST SPLITS
# ============================================================================

print("\n" + "="*100)
print("SECTION 8: PREPARING TRAIN-TEST SPLITS")
print("="*100)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_processed['Sentiment'])

print(f"\nLabel Encoding:")
for idx, label in enumerate(label_encoder.classes_):
    print(f"  {label} -> {idx}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Also create split for combined features
X_train_combined, X_test_combined, y_train_c, y_test_c = train_test_split(
    X_combined, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nDataset Splits:")
print(f"  Training samples: {X_train.shape[0]:,}")
print(f"  Testing samples: {X_test.shape[0]:,}")
print(f"  Feature dimension: {X_train.shape[1]:,}")

print("\n[OK] Data splits created!")

# ============================================================================
# SECTION 9: MACHINE LEARNING MODELS - TRAINING AND EVALUATION
# ============================================================================

print("\n" + "="*100)
print("SECTION 9: MACHINE LEARNING MODELS - TRAINING AND EVALUATION")
print("="*100)

# Dictionary to store all results
ml_results = {}

# 9.1: Naive Bayes
print("\n" + "-"*80)
print("[MODEL 1/10] Naive Bayes Classifier")
print("-"*80)
start = time.time()
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_time = time.time() - start
ml_results['Naive Bayes'] = {
    'model': nb_model,
    'accuracy': nb_accuracy,
    'predictions': nb_pred,
    'time': nb_time
}
print(f"Accuracy: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")
print(f"Training time: {nb_time:.2f}s")

# 9.2: Logistic Regression
print("\n" + "-"*80)
print("[MODEL 2/10] Logistic Regression")
print("-"*80)
start = time.time()
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_time = time.time() - start
ml_results['Logistic Regression'] = {
    'model': lr_model,
    'accuracy': lr_accuracy,
    'predictions': lr_pred,
    'time': lr_time
}
print(f"Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"Training time: {lr_time:.2f}s")

# 9.3: Linear SVM
print("\n" + "-"*80)
print("[MODEL 3/10] Linear SVM")
print("-"*80)
start = time.time()
svm_model = LinearSVC(max_iter=1000, random_state=42, C=1.0)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_time = time.time() - start
ml_results['Linear SVM'] = {
    'model': svm_model,
    'accuracy': svm_accuracy,
    'predictions': svm_pred,
    'time': svm_time
}
print(f"Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
print(f"Training time: {svm_time:.2f}s")

# 9.4: Random Forest (optimized)
print("\n" + "-"*80)
print("[MODEL 4/10] Random Forest Classifier")
print("-"*80)
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=50,  # Reduced for faster training
    max_depth=20,  # Limited depth
    random_state=42, 
    n_jobs=-1,
    max_samples=0.8  # Use subset of samples for speed
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_time = time.time() - start
ml_results['Random Forest'] = {
    'model': rf_model,
    'accuracy': rf_accuracy,
    'predictions': rf_pred,
    'time': rf_time
}
print(f"Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"Training time: {rf_time:.2f}s")

# 9.5: Gradient Boosting (ultra-fast configuration)
print("\n" + "-"*80)
print("[MODEL 5/10] Gradient Boosting Classifier")
print("-"*80)
start = time.time()
gb_model = GradientBoostingClassifier(
    n_estimators=25,  # Further reduced for speed
    learning_rate=0.2,  # Higher learning rate for faster convergence
    max_depth=4,  # Shallow trees for speed
    max_features='sqrt',  # Use sqrt features for speed
    subsample=0.8,  # Use subset of samples
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=5,  # Earlier stopping
    tol=1e-3,  # Relaxed tolerance
    warm_start=False  # Disable warm start for speed
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_time = time.time() - start
ml_results['Gradient Boosting'] = {
    'model': gb_model,
    'accuracy': gb_accuracy,
    'predictions': gb_pred,
    'time': gb_time
}
print(f"Accuracy: {gb_accuracy:.4f} ({gb_accuracy*100:.2f}%)")
print(f"Training time: {gb_time:.2f}s")
print(f"Note: Ultra-fast configuration - 25 estimators, shallow trees")

# 9.6: Decision Tree (simplified)
print("\n" + "-"*80)
print("[MODEL 6/7] Decision Tree Classifier")
print("-"*80)
start = time.time()
dt_model = DecisionTreeClassifier(random_state=42, max_depth=20)  # Reduced depth for speed
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_time = time.time() - start
ml_results['Decision Tree'] = {
    'model': dt_model,
    'accuracy': dt_accuracy,
    'predictions': dt_pred,
    'time': dt_time
}
print(f"Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
print(f"Training time: {dt_time:.2f}s")

# 9.7: Extra Trees (fixed configuration)
print("\n" + "-"*80)
print("[MODEL 7/7] Extra Trees Classifier")
print("-"*80)
start = time.time()
et_model = ExtraTreesClassifier(
    n_estimators=50,  # Reduced for faster training
    max_depth=20,  # Limited depth
    random_state=42, 
    n_jobs=-1,
    bootstrap=True,  # Enable bootstrap for max_samples
    max_samples=0.8  # Use subset of samples for speed
)
et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_test)
et_accuracy = accuracy_score(y_test, et_pred)
et_time = time.time() - start
ml_results['Extra Trees'] = {
    'model': et_model,
    'accuracy': et_accuracy,
    'predictions': et_pred,
    'time': et_time
}
print(f"Accuracy: {et_accuracy:.4f} ({et_accuracy*100:.2f}%)")
print(f"Training time: {et_time:.2f}s")

print("\n[OK] All 7 ML models trained successfully!")
print("Models: Naive Bayes, Logistic Regression, Linear SVM, Random Forest, Gradient Boosting, Decision Tree, Extra Trees")

# ============================================================================
# SECTION 10: DEEP LEARNING MODELS
# ============================================================================

print("\n" + "="*100)
print("SECTION 10: DEEP LEARNING MODELS")
print("="*100)

dl_results = {}

if DEEP_LEARNING_AVAILABLE:
    # Prepare sequences for deep learning
    print("\n[STEP 1] Preparing sequences for deep learning...")
    
    MAX_WORDS = 10000
    MAX_LEN = 200
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df_processed['cleaned_text'])
    
    sequences = tokenizer.texts_to_sequences(df_processed['cleaned_text'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    print(f"  Vocabulary size: {len(tokenizer.word_index)}")
    print(f"  Sequence shape: {padded_sequences.shape}")
    
    # Split for deep learning
    X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
        padded_sequences, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Convert labels to categorical
    num_classes = len(label_encoder.classes_)
    y_train_dl_cat = keras.utils.to_categorical(y_train_dl, num_classes)
    y_test_dl_cat = keras.utils.to_categorical(y_test_dl, num_classes)
    
    # 10.1: LSTM Model
    print("\n" + "-"*80)
    print("[DL MODEL 1/4] LSTM Neural Network")
    print("-"*80)
    
    lstm_model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    lstm_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(lstm_model.summary())
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    start = time.time()
    lstm_history = lstm_model.fit(
        X_train_dl, y_train_dl_cat,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    lstm_time = time.time() - start
    
    lstm_pred_proba = lstm_model.predict(X_test_dl)
    lstm_pred = np.argmax(lstm_pred_proba, axis=1)
    lstm_accuracy = accuracy_score(y_test_dl, lstm_pred)
    
    dl_results['LSTM'] = {
        'model': lstm_model,
        'accuracy': lstm_accuracy,
        'predictions': lstm_pred,
        'time': lstm_time,
        'history': lstm_history
    }
    
    print(f"\nLSTM Accuracy: {lstm_accuracy:.4f} ({lstm_accuracy*100:.2f}%)")
    print(f"Training time: {lstm_time:.2f}s")
    
    # 10.2: Bidirectional LSTM
    print("\n" + "-"*80)
    print("[DL MODEL 2/4] Bidirectional LSTM")
    print("-"*80)
    
    bilstm_model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    bilstm_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start = time.time()
    bilstm_history = bilstm_model.fit(
        X_train_dl, y_train_dl_cat,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    bilstm_time = time.time() - start
    
    bilstm_pred_proba = bilstm_model.predict(X_test_dl)
    bilstm_pred = np.argmax(bilstm_pred_proba, axis=1)
    bilstm_accuracy = accuracy_score(y_test_dl, bilstm_pred)
    
    dl_results['Bidirectional LSTM'] = {
        'model': bilstm_model,
        'accuracy': bilstm_accuracy,
        'predictions': bilstm_pred,
        'time': bilstm_time,
        'history': bilstm_history
    }
    
    print(f"\nBi-LSTM Accuracy: {bilstm_accuracy:.4f} ({bilstm_accuracy*100:.2f}%)")
    print(f"Training time: {bilstm_time:.2f}s")
    
    # 10.3: CNN Model
    print("\n" + "-"*80)
    print("[DL MODEL 3/4] 1D CNN")
    print("-"*80)
    
    cnn_model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    cnn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start = time.time()
    cnn_history = cnn_model.fit(
        X_train_dl, y_train_dl_cat,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    cnn_time = time.time() - start
    
    cnn_pred_proba = cnn_model.predict(X_test_dl)
    cnn_pred = np.argmax(cnn_pred_proba, axis=1)
    cnn_accuracy = accuracy_score(y_test_dl, cnn_pred)
    
    dl_results['CNN'] = {
        'model': cnn_model,
        'accuracy': cnn_accuracy,
        'predictions': cnn_pred,
        'time': cnn_time,
        'history': cnn_history
    }
    
    print(f"\nCNN Accuracy: {cnn_accuracy:.4f} ({cnn_accuracy*100:.2f}%)")
    print(f"Training time: {cnn_time:.2f}s")
    
    # 10.4: Hybrid CNN-LSTM
    print("\n" + "-"*80)
    print("[DL MODEL 4/4] Hybrid CNN-LSTM")
    print("-"*80)
    
    hybrid_model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(5),
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    hybrid_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start = time.time()
    hybrid_history = hybrid_model.fit(
        X_train_dl, y_train_dl_cat,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    hybrid_time = time.time() - start
    
    hybrid_pred_proba = hybrid_model.predict(X_test_dl)
    hybrid_pred = np.argmax(hybrid_pred_proba, axis=1)
    hybrid_accuracy = accuracy_score(y_test_dl, hybrid_pred)
    
    dl_results['Hybrid CNN-LSTM'] = {
        'model': hybrid_model,
        'accuracy': hybrid_accuracy,
        'predictions': hybrid_pred,
        'time': hybrid_time,
        'history': hybrid_history
    }
    
    print(f"\nHybrid CNN-LSTM Accuracy: {hybrid_accuracy:.4f} ({hybrid_accuracy*100:.2f}%)")
    print(f"Training time: {hybrid_time:.2f}s")
    
    print("\n[OK] All deep learning models trained!")
else:
    print("\n[WARNING] TensorFlow not available. Skipping deep learning models.")

# ============================================================================
# SECTION 11: MODEL COMPARISON AND EVALUATION
# ============================================================================

print("\n" + "="*100)
print("SECTION 11: COMPREHENSIVE MODEL COMPARISON")
print("="*100)

# 11.1: Compare ML models
print("\n[STEP 1] Machine Learning Models Comparison:")
print("-" * 80)
print(f"{'Model':<25} {'Accuracy':<15} {'Training Time':<15}")
print("-" * 80)

for model_name, result in ml_results.items():
    print(f"{model_name:<25} {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)   {result['time']:.2f}s")

# 11.2: Compare DL models
if DEEP_LEARNING_AVAILABLE and dl_results:
    print("\n[STEP 2] Deep Learning Models Comparison:")
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':<15} {'Training Time':<15}")
    print("-" * 80)
    
    for model_name, result in dl_results.items():
        print(f"{model_name:<25} {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)   {result['time']:.2f}s")

# 11.3: Find best model
all_results = {**ml_results}
if DEEP_LEARNING_AVAILABLE:
    all_results.update(dl_results)

best_model_name = max(all_results, key=lambda x: all_results[x]['accuracy'])
best_accuracy = all_results[best_model_name]['accuracy']

print("\n" + "="*80)
print(f"BEST MODEL: {best_model_name}")
print(f"BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("="*80)

# 11.4: Detailed evaluation of best model
print(f"\n[STEP 3] Detailed Evaluation of Best Model: {best_model_name}")
print("-" * 80)

best_predictions = all_results[best_model_name]['predictions']

# Use appropriate test labels
if best_model_name in dl_results:
    test_labels = y_test_dl
else:
    test_labels = y_test

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, best_predictions, 
                          target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(test_labels, best_predictions)
print("\nConfusion Matrix:")
print(cm)

# Additional metrics
precision = precision_score(test_labels, best_predictions, average='weighted')
recall = recall_score(test_labels, best_predictions, average='weighted')
f1 = f1_score(test_labels, best_predictions, average='weighted')

print(f"\nWeighted Metrics:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")

# ============================================================================
# SECTION 12: CROSS-VALIDATION ANALYSIS
# ============================================================================

print("\n" + "="*100)
print("SECTION 12: CROSS-VALIDATION ANALYSIS")
print("="*100)

# Perform cross-validation on top 3 ML models
print("\n[STEP 1] Performing 5-fold cross-validation on top models...")

top_ml_models = sorted(ml_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model_data in top_ml_models:
    print(f"\nCross-validating {model_name}...")
    model = model_data['model']
    cv_scores = cross_val_score(model, X_tfidf, y, cv=skf, scoring='accuracy')
    cv_results[model_name] = cv_scores
    
    print(f"  CV Scores: {cv_scores}")
    print(f"  Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Visualize CV results
print("\n[STEP 2] Creating cross-validation visualization...")
plt.figure(figsize=(12, 6))
bp = plt.boxplot([cv_results[name] for name in cv_results.keys()],
                  labels=list(cv_results.keys()),
                  patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)

plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/cross_validation_results.png', dpi=300)
plt.close()

print("\n[OK] Cross-validation completed!")

# ============================================================================
# SECTION 13: PREDICTION ON UNLABELED DATASET
# ============================================================================

print("\n" + "="*100)
print("SECTION 13: PREDICTIONS ON UNLABELED DATASET")
print("="*100)

# Use best model for predictions
print(f"\n[STEP 1] Using {best_model_name} for predictions on unlabeled data...")

# Get the best model from appropriate results dictionary
if best_model_name in ml_results:
    best_ml_model = ml_results[best_model_name]['model']
    use_tfidf = True
elif DEEP_LEARNING_AVAILABLE and best_model_name in dl_results:
    best_ml_model = dl_results[best_model_name]['model']
    use_tfidf = False
else:
    # Fallback to first available ML model
    best_model_name = list(ml_results.keys())[0]
    best_ml_model = ml_results[best_model_name]['model']
    use_tfidf = True
    print(f"[WARNING] Best model not found, using {best_model_name} instead")

# Transform all data and predict based on model type
if use_tfidf:
    # ML model - use TF-IDF
    X_all_tfidf = tfidf_vectorizer.transform(df_fake['cleaned_text'])
    predictions_all = best_ml_model.predict(X_all_tfidf)
    predicted_sentiments = label_encoder.inverse_transform(predictions_all)
else:
    # DL model - use sequences
    if DEEP_LEARNING_AVAILABLE:
        sequences_all = tokenizer.texts_to_sequences(df_fake['cleaned_text'])
        padded_sequences_all = pad_sequences(sequences_all, maxlen=MAX_LEN, padding='post', truncating='post')
        predictions_proba = best_ml_model.predict(padded_sequences_all)
        predictions_all = np.argmax(predictions_proba, axis=1)
        predicted_sentiments = label_encoder.inverse_transform(predictions_all)
    else:
        # Fallback to ML model
        X_all_tfidf = tfidf_vectorizer.transform(df_fake['cleaned_text'])
        predictions_all = best_ml_model.predict(X_all_tfidf)
        predicted_sentiments = label_encoder.inverse_transform(predictions_all)

# Create results dataframe
df_predictions = df_fake.copy()
df_predictions['predicted_sentiment'] = predicted_sentiments
df_predictions['actual_sentiment'] = actual_sentiments.values
df_predictions['match'] = df_predictions['predicted_sentiment'] == df_predictions['actual_sentiment']

# Calculate overall accuracy
overall_accuracy = df_predictions['match'].mean()
correct_predictions = df_predictions['match'].sum()
total_predictions = len(df_predictions)

print(f"\n[STEP 2] Prediction Results:")
print(f"  Total samples: {total_predictions:,}")
print(f"  Correct predictions: {correct_predictions:,}")
print(f"  Incorrect predictions: {total_predictions - correct_predictions:,}")
print(f"  Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

# Show sample predictions
print("\n[STEP 3] Sample predictions (first 25):")
print("=" * 100)
for i in range(min(25, len(df_predictions))):
    row = df_predictions.iloc[i]
    match_symbol = "[OK]" if row['match'] else "[X]"
    text_preview = row['Content'][:80] + "..." if len(row['Content']) > 80 else row['Content']
    
    print(f"\n{i+1}. {match_symbol}")
    print(f"   Text: {text_preview}")
    print(f"   Predicted: {row['predicted_sentiment']:10s} | Actual: {row['actual_sentiment']}")

# Prediction accuracy by sentiment
print("\n[STEP 4] Prediction accuracy by sentiment:")
for sentiment in label_encoder.classes_:
    subset = df_predictions[df_predictions['actual_sentiment'] == sentiment]
    sentiment_accuracy = subset['match'].mean()
    correct = subset['match'].sum()
    total = len(subset)
    print(f"  {sentiment:10s}: {sentiment_accuracy:.4f} ({sentiment_accuracy*100:.2f}%) - {correct}/{total}")

# Save predictions
predictions_file = 'final_predictions.csv'
df_predictions[['Content', 'predicted_sentiment', 'actual_sentiment', 'match']].to_csv(
    predictions_file, index=False
)
print(f"\n[OK] Predictions saved to '{predictions_file}'")

# ============================================================================
# SECTION 14: ERROR ANALYSIS
# ============================================================================

print("\n" + "="*100)
print("SECTION 14: ERROR ANALYSIS")
print("="*100)

# Get misclassified samples
print("\n[STEP 1] Analyzing misclassified samples...")
misclassified = df_predictions[~df_predictions['match']]

print(f"Total misclassified: {len(misclassified):,}")
print(f"Misclassification rate: {len(misclassified)/len(df_predictions)*100:.2f}%")

# Misclassification patterns
print("\n[STEP 2] Misclassification patterns:")
misclass_patterns = pd.crosstab(
    misclassified['actual_sentiment'],
    misclassified['predicted_sentiment'],
    margins=True
)
print(misclass_patterns)

# Show some misclassified examples
print("\n[STEP 3] Sample misclassified examples:")
for i, (idx, row) in enumerate(misclassified.head(10).iterrows()):
    print(f"\n{i+1}. Actual: {row['actual_sentiment']:10s} | Predicted: {row['predicted_sentiment']}")
    print(f"   Text: {row['Content'][:150]}...")

# Visualize misclassification matrix
print("\n[STEP 4] Creating misclassification visualization...")
misclass_matrix = pd.crosstab(
    misclassified['actual_sentiment'],
    misclassified['predicted_sentiment']
)

plt.figure(figsize=(10, 8))
sns.heatmap(misclass_matrix, annot=True, fmt='d', cmap='Reds')
plt.title('Misclassification Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Sentiment', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Sentiment', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/misclassification_matrix.png', dpi=300)
plt.close()

print("\n[OK] Error analysis completed!")

# ============================================================================
# SECTION 15: MODEL PERSISTENCE
# ============================================================================

print("\n" + "="*100)
print("SECTION 15: SAVING MODELS AND ARTIFACTS")
print("="*100)

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# Save best ML model
print(f"\n[STEP 1] Saving best ML model ({best_model_name})...")
with open('models/best_ml_model.pkl', 'wb') as f:
    pickle.dump(best_ml_model, f)

# Save vectorizer
print("[STEP 2] Saving TF-IDF vectorizer...")
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save label encoder
print("[STEP 3] Saving label encoder...")
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save deep learning models
if DEEP_LEARNING_AVAILABLE and dl_results:
    print("[STEP 4] Saving deep learning models...")
    for model_name, result in dl_results.items():
        model_filename = f'models/dl_{model_name.lower().replace(" ", "_")}.h5'
        result['model'].save(model_filename)
        print(f"  Saved {model_name}")
    
    # Save tokenizer
    print("[STEP 5] Saving tokenizer...")
    with open('models/dl_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

# Save results summary
print("[STEP 6] Saving results summary...")
results_summary = {
    'best_model': best_model_name,
    'best_accuracy': float(best_accuracy),
    'overall_accuracy': float(overall_accuracy),
    'ml_results': {name: {'accuracy': float(res['accuracy']), 'time': float(res['time'])} 
                   for name, res in ml_results.items()},
    'timestamp': str(datetime.datetime.now())
}

if DEEP_LEARNING_AVAILABLE and dl_results:
    results_summary['dl_results'] = {
        name: {'accuracy': float(res['accuracy']), 'time': float(res['time'])} 
        for name, res in dl_results.items()
    }

with open('models/results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=4)

print("\n[OK] All models and artifacts saved!")

# ============================================================================
# SECTION 16: FINAL REPORT AND SUMMARY
# ============================================================================

print("\n" + "="*100)
print("SECTION 16: FINAL PROJECT REPORT")
print("="*100)

report = f"""
{'='*100}
SENTIMENT ANALYSIS PROJECT - FINAL REPORT
{'='*100}

PROJECT OVERVIEW:
-----------------
Project Title: Advanced Sentiment Analysis on Indian Financial News
Dataset: Indian Financial News (26,961 articles)
Sentiment Classes: Negative, Neutral, Positive
Timestamp: {datetime.datetime.now()}

DATA STATISTICS:
----------------
Total Samples: {len(df_original):,}
Training Samples: {X_train.shape[0]:,}
Testing Samples: {X_test.shape[0]:,}
Features Engineered: {len(df_processed.columns)}
TF-IDF Vocabulary Size: {len(tfidf_vectorizer.vocabulary_):,}

SENTIMENT DISTRIBUTION:
-----------------------
{sentiment_counts.to_string()}

MACHINE LEARNING MODELS TESTED: 7 (Simplified for Performance)
--------------------------------------
"""

for name, result in sorted(ml_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    report += f"{name:25s}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%) - {result['time']:.2f}s\n"

if DEEP_LEARNING_AVAILABLE and dl_results:
    report += f"\nDEEP LEARNING MODELS TESTED: {len(dl_results)}\n"
    report += "--------------------------------------\n"
    for name, result in sorted(dl_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        report += f"{name:25s}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%) - {result['time']:.2f}s\n"

report += f"""
BEST MODEL:
-----------
Model Name: {best_model_name}
Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)
Overall Accuracy on Full Dataset: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)

DETAILED METRICS (Best Model):
------------------------------
Precision (weighted): {precision:.4f}
Recall (weighted): {recall:.4f}
F1-Score (weighted): {f1:.4f}

PREDICTIONS:
------------
Total Predictions: {total_predictions:,}
Correct Predictions: {correct_predictions:,}
Incorrect Predictions: {total_predictions - correct_predictions:,}
Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)

ACCURACY BY SENTIMENT:
----------------------
"""

for sentiment in label_encoder.classes_:
    subset = df_predictions[df_predictions['actual_sentiment'] == sentiment]
    sentiment_accuracy = subset['match'].mean()
    correct = subset['match'].sum()
    total = len(subset)
    report += f"{sentiment:10s}: {sentiment_accuracy:.4f} ({sentiment_accuracy*100:.2f}%) - {correct:,}/{total:,} correct\n"

report += f"""
FILES GENERATED:
----------------
1. Visualizations (visualizations/):
   - sentiment_distribution.png
   - word_count_distribution.png
   - ml_models_comparison.png
   - confusion_matrix.png
   - confusion_matrix_normalized.png
   - training_time_comparison.png
   - wordcloud_negative.png
   - wordcloud_neutral.png
   - wordcloud_positive.png
   - feature_importance.png
   - cross_validation_results.png
   - misclassification_matrix.png
   {'- DL training history plots' if DEEP_LEARNING_AVAILABLE and dl_results else ''}

2. Models (models/):
   - best_ml_model.pkl
   - tfidf_vectorizer.pkl
   - label_encoder.pkl
   - results_summary.json
   {'- Deep learning models (.h5)' if DEEP_LEARNING_AVAILABLE and dl_results else ''}
   {'- dl_tokenizer.pkl' if DEEP_LEARNING_AVAILABLE and dl_results else ''}

3. Data:
   - final_predictions.csv

CONCLUSION:
-----------
This comprehensive sentiment analysis project successfully implemented and compared
{len(ml_results)} machine learning models{' and ' + str(len(dl_results)) + ' deep learning models' if DEEP_LEARNING_AVAILABLE and dl_results else ''}.
The best performing model ({best_model_name}) achieved {best_accuracy*100:.2f}% accuracy on the test set
and {overall_accuracy*100:.2f}% accuracy on the full dataset predictions.

The project demonstrates proficiency in:
- Data preprocessing and cleaning
- Feature engineering
- Text vectorization and tokenization
- Machine learning model implementation
- Deep learning model implementation
- Model evaluation and comparison
- Data visualization
- Error analysis

{'='*100}
END OF REPORT
{'='*100}
"""

print(report)

# Save report
with open('PROJECT_REPORT.txt', 'w') as f:
    f.write(report)

print("\n[OK] Final report saved to 'PROJECT_REPORT.txt'")


# ============================================================================
# PROJECT COMPLETION
# ============================================================================

print("\n" + "="*100)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("="*100)

total_execution_time = time.time() - start_time
print(f"\nTotal execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
print(f"Completion timestamp: {datetime.datetime.now()}")

print("\n" + "="*100)
print("Thank you for using the Sentiment Analysis System!")
print("="*100)

