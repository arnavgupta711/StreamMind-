import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import re
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')  # This downloads the missing 'punkt_tab' tokenizer
nltk.download('stopwords')

# Load the dataset with the correct file path
data_path = r"C:\Users\HP\Favorites\Downloads\archive (3)\twitch_reviews.csv"  # Raw string for Windows paths
df = pd.read_csv(data_path)

# Clean and preprocess the reviews
def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()  # Convert to lowercase
    return text

# Apply text cleaning
df['cleaned_content'] = df['content'].apply(clean_text)

# Tokenize the text and remove stopwords
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

df['tokens'] = df['cleaned_content'].apply(tokenize_and_remove_stopwords)

# Check for any missing values
print(df.isnull().sum())

# Split the dataset into train and test sets (80-20 split)
X = df['tokens']  # Features (tokens)
y = df['reviewId']  # Target (Sentiment labels or satisfaction levels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure that the 'data' directory exists
if not os.path.exists('../data'):
    os.makedirs('../data')

# Save processed data for future use
df.to_csv(r'../data/processed_twitch_reviews.csv', index=False)  # Save the processed file

print("Data preprocessing complete!")
