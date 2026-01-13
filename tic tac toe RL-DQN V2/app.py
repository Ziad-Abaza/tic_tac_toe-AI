from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app) 

try:
    model = tf.keras.models.load_model('tic_tac_toe.h5')
except:
    from tensorflow.keras.losses import MeanSquaredError
    model = tf.keras.models.load_model('tic_tac_toe.h5', custom_objects={'mse': MeanSquaredError()})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_board = data['board'] 
    current_turn = data.get('turn', 1) 

    standardized_board = np.array(raw_board) * current_turn
    input_board = standardized_board.reshape(1, 3, 3, 1)
    predictions = model.predict(input_board, verbose=0)[0]
    sorted_moves = np.argsort(predictions)[::-1]
    
    next_move = None
    for move in sorted_moves:
        if raw_board[move] == 0: 
            next_move = move
            break
            
    if next_move is None:
        next_move = raw_board.index(0)

    print(f"Turn: {current_turn}, Best Predicted Values: {predictions[next_move]}, Move: {next_move}")
    
    return jsonify({
        'next_move': int(next_move),
        'confidence': float(predictions[next_move])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)