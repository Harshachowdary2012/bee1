"""
Simple Sentiment Analysis on Indian Financial News
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
ds = load_dataset("kdave/Indian_Financial_News")

# Original dataset with labels
df_original = pd.DataFrame(ds['train'])
print(f"\n[OK] Original dataset with labels: {df_original.shape}")
print(f"Columns: {df_original.columns.tolist()}")
print(f"\nSentiment distribution:")
print(df_original['Sentiment'].value_counts())

# Create a fake copy WITHOUT labels
df_fake = df_original.drop('Sentiment', axis=1).copy()
print(f"\n[OK] Fake dataset without labels: {df_fake.shape}")
print(f"Columns: {df_fake.columns.tolist()}")

# Store actual sentiments separately for comparison later
actual_sentiments = df_original['Sentiment'].copy()

print("\n" + "="*80)
print("TRAINING MODEL TO PREDICT SENTIMENTS")
print("="*80)

# Prepare data for training
X = df_original['Content']  # The text column is 'Content'
y = df_original['Sentiment']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Tokenization using TF-IDF
print("\nTokenizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)} words")

# Train model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Test accuracy
y_pred = model.predict(X_test_vec)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n[OK] Model trained successfully!")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n" + "="*80)
print("PREDICTING ON FAKE DATASET (WITHOUT LABELS)")
print("="*80)

# Now predict on the fake dataset
X_fake_vec = vectorizer.transform(df_fake['Content'])
predicted_sentiments = model.predict(X_fake_vec)

# Add predictions to fake dataset
df_fake['predicted_sentiment'] = predicted_sentiments
df_fake['actual_sentiment'] = actual_sentiments.values

# Compare predictions with actual
print("\nComparison of Predictions vs Actual Labels:")
print("-" * 80)

print("\nFirst 20 samples:")
for i in range(20):
    text = df_fake['Content'].iloc[i]
    pred = df_fake['predicted_sentiment'].iloc[i]
    actual = df_fake['actual_sentiment'].iloc[i]
    match = "[OK]" if pred == actual else "[X]"
    
    print(f"\n{i+1}. {match} Text: {text[:80]}...")
    print(f"   Predicted: {pred:10s} | Actual: {actual}")

# Overall accuracy
correct = (df_fake['predicted_sentiment'] == df_fake['actual_sentiment']).sum()
total = len(df_fake)
overall_accuracy = correct / total

print("\n" + "="*80)
print(f"RESULTS:")
print(f"  Correct Predictions: {correct}/{total}")
print(f"  Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print("="*80)

# Save results
df_fake.to_csv('predictions_comparison.csv', index=False)
print("\n[OK] Results saved to 'predictions_comparison.csv'")

print("\n" + "="*80)
print("DONE!")
print("="*80)

