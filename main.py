# main.py
"""
Very minimal flask app for hosting my model
""""

from flask import Flask, request, render_template, redirect, url_for, jsonify
import joblib


app = Flask(__name__)


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
        
        if request.form:
            text = request.form.get('text', '').strip()
        else:
           
            data = request.json
            text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided. Please include a "text" key in your JSON payload.'}), 400

        
        X = tfidf.transform([text])
       
        prediction = model.predict(X)

        sentiment = label_encoder.inverse_transform(prediction)[0]

        
        if request.form:
            return render_template('result.html', text=text, sentiment=sentiment)

        
        return jsonify({'text': text, 'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False)
