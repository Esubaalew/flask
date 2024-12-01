from flask import Flask, request, jsonify
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained components
model = joblib.load('models/sentiment_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API!"

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        # Get the input data (JSON format expected)
        data = request.json

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided. Please include a "text" key in your JSON payload.'}), 400

        text = data['text']

        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Invalid text input. It must be a non-empty string.'}), 400

        # Preprocess and transform the input text
        X = tfidf.transform([text])

        # Predict sentiment
        prediction = model.predict(X)

        # Decode the predicted sentiment label
        sentiment = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'text': text, 'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
