from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model, vectorizer, and label encoder
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Download NLTK resources
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = text.split()  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve email content from the form
        email_content = request.form.get('email')
        if not email_content:
            return jsonify({'error': 'No email content provided'}), 400

        # Preprocess the email content
        cleaned_message = preprocess_text(email_content)

        # Transform the email content using the vectorizer
        email_vector = vectorizer.transform([cleaned_message])

        # Predict spam or not
        prediction = model.predict(email_vector)[0]
        decoded_prediction = encoder.inverse_transform([prediction])[0]

        # Return the result as JSON
        return jsonify({'prediction': decoded_prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
