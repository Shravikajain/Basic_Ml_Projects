import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Download NLTK resources
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load your dataset
data = pd.read_csv('C:/Users/shrav/Downloads/archive(13)/email.csv')  # Replace with your dataset path

# Clean the data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = text.split()  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Use the 'Message' column for email content
data['cleaned_message'] = data['Message'].apply(preprocess_text)

# Encode labels (if 'category' is not already numeric)
encoder = LabelEncoder()
data['encoded_category'] = encoder.fit_transform(data['Category'])  # e.g., spam -> 1, not spam -> 0

# Feature extraction
vectorizer = CountVectorizer()
X_vectors = vectorizer.fit_transform(data['cleaned_message'])
y = data['encoded_category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model, vectorizer, and encoder
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(encoder, encoder_file)
