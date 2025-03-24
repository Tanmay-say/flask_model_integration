from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = 'lstm_model.h5'  # Update with your model path
TOKENIZER_PATH = 'tokenizer.pkl'  # Update with your tokenizer path

# Check if model and tokenizer files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    print(f"Warning: Model file ({MODEL_PATH}) or tokenizer file ({TOKENIZER_PATH}) not found.")
    print("Please place your model.h5 and tokenizer.pkl files in the root directory.")
    model = None
    tokenizer = None
else:
    try:
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)

        # Load the tokenizer
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)

        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        model = None
        tokenizer = None


def predict_grade(question, desired_answer, student_answer):
    """
    Process inputs through the model and return a grade between 1-5.
    Modify this function according to your model's specific input requirements.
    """
    if model is None or tokenizer is None:
        return "Error: Model or tokenizer not loaded"

    try:
        # This is a placeholder implementation - modify according to your model's requirements
        # For example, you might need to tokenize and pad the inputs

        # Example preprocessing (adjust based on your model's requirements)
        inputs = [question, desired_answer, student_answer]
        # Tokenize inputs
        tokenized_inputs = [tokenizer.texts_to_sequences([text])[0] for text in inputs]
        # Pad sequences if needed
        # padded_inputs = pad_sequences(tokenized_inputs, maxlen=MAX_LENGTH)

        # Make prediction
        prediction = model.predict(np.array(tokenized_inputs))

        # Convert prediction to grade (1-5)
        # This is just an example - adjust based on your model's output format
        grade = np.clip(round(float(prediction[0][0])), 1, 5)

        return grade
    except Exception as e:
        return f"Error during prediction: {e}"


@app.route('/', methods=['GET', 'POST'])
def index():
    grade = None
    if request.method == 'POST':
        question = request.form.get('question', '')
        desired_answer = request.form.get('desired_answer', '')
        student_answer = request.form.get('student_answer', '')

        if question and desired_answer and student_answer:
            grade = predict_grade(question, desired_answer, student_answer)

    return render_template('index.html', grade=grade)


@app.route('/api/grade', methods=['POST'])
def api_grade():
    data = request.json
    question = data.get('question', '')
    desired_answer = data.get('desired_answer', '')
    student_answer = data.get('student_answer', '')

    if question and desired_answer and student_answer:
        grade = predict_grade(question, desired_answer, student_answer)
        return jsonify({'grade': grade})
    else:
        return jsonify({'error': 'Missing required fields'}), 400


if __name__ == '__main__':
    app.run(debug=True)