# uses pandas, scikitlearn, flask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import MiniBatchKMeans
from flask import Flask,request,jsonify,send_file

class ImageTransformer(object):
    @staticmethod
    def transform(image_as_array, n_clusters):
        (h,w,c) = image_as_array.shape
        print(f'Image parameters: (h: {h}, w: {w}, c: {c})')
        image_as_array2d = image_as_array.reshape(h*w,c)
        model = MiniBatchKMeans(n_clusters=n_clusters)
        labels = model.fit_predict(image_as_array2d)
        rgb_codes = model.cluster_centers_.round(0).astype(np.uint8)
        quantized_image = np.reshape(rgb_codes[labels],(h,w,c))
        plt.imshow(quantized_image)
        return quantized_image

# Create Flask APP
app = Flask(__name__)

# Connect POST API call to function
@app.route('/transform',methods=['POST'])
def transform():
    f = request.files.get('file')
    image_as_array = plt.imread(f)
    # Get data from request
    # feature_date = request.json
    #
    # Convert it to image array
    # df = pd.DataFrame(feature_data)
    # df = df.reindex(columns=col_names) # imported column names
    # Transform and return image
    transformed_image = ImageTransformer().transform(image_as_array,10)
    # return jsonify({'prediction':str(prediction)})
    plt.imsave('img.jpg',transformed_image)
    return send_file('img.jpg')

# Load model
if __name__ == "__main__":
    #image_transformer = joblib.load('image_transformer.pkl')
    app.run(debug=True)
