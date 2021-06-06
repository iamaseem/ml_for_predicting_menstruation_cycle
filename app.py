from flask import Flask, request, jsonify
import joblib

# Define the app
app = Flask(__name__)
classifier = joblib.load('cat_clf.pkl')

# Get a welcoming message once you start the server.
@app.route('/predict', methods=['POST'])
def login():
  values = request.get_json()
  arr = [values['age'],values['weight'],values['height'],values['bmi'],11,72,22,12.0,2,4,15.0,0,0,10.00,10.00,5.01,6.48,0.773148,40,37,0.925000,4.21,1.89,12.71,32.45,0.27,85.0,1,1,1,1,1,1.0,0,110,80,5,10,14.0,13.0,6.0]
  prediction = classifier.predict(arr)
  return str(prediction)

# If the file is run directly,start the app.
if __name__ == '__main__':
    app.run()