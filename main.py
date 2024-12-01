from flask import Flask, request, render_template, redirect, url_for, jsonify
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained components
model = joblib.load('models/sentiment_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

@app.route('/')
def home():
    """Render the home page with a form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """Handle sentiment prediction via form or API."""
    try:
        # Handle form submission
        if request.form:
            text = request.form.get('text', '').strip()
        else:
            # Handle API request
            data = request.json
            text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided. Please include a "text" key in your JSON payload.'}), 400

        # Preprocess and transform the input text
        X = tfidf.transform([text])

        # Predict sentiment
        prediction = model.predict(X)

        # Decode the predicted sentiment label
        sentiment = label_encoder.inverse_transform(prediction)[0]

        # Return result for form submission
        if request.form:
            return render_template('result.html', text=text, sentiment=sentiment)

        # Return result for API request
        return jsonify({'text': text, 'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False)
