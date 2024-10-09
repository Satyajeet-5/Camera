from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    # Retrieve the images from the request
    image1 = request.files['image1']
    image2 = request.files['image2']

    # Save the images locally (or process directly if preferred)
    image1.save("image1.jpg")
    image2.save("image2.jpg")

    # Perform face comparison
    result = DeepFace.verify("image1.jpg", "image2.jpg", model_name="Facenet")
    is_match = result['verified']

    # Send the result back to the PythonAnywhere app
    return jsonify(result="Match" if is_match else "No Match")

if __name__ == '__main__':
    app.run()