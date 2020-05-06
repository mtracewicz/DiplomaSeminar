// Import Tensorflow JS
importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.2.7/dist/tf.min.js"
);

// Make prediction and send back answer
onmessage = async (e) => {
  tf.setBackend("cpu");
  const input = getInputTensor(preprocessImage(e.data));
  const model = await tf.loadLayersModel("model/model.json");
  const prediction_array = await model.predict(input).data();
  postMessage(getPredictionFromArray(prediction_array));
};

// Convert data into tensor which can be used as tensorflow model input
function getInputTensor(data) {
  return tf.tensor(data).asType("float32").reshape([1, 28, 28, 1]);
}

// Prepare data to fit model input
function preprocessImage(image) {
  return makeMonochrome(scaleData(image));
}

// Scale data from integers [0,255] to floats [0.0,1.0]
function scaleData(data) {
  return new Float32Array(data).map((i) => i / 255.0);
}

/*
  Get monochrome version from RGBA
  We can assign red color from input data to processed data,
  because as the pictures are monochrome in their nature values of 
  red,green,blue are always the same.
*/
function makeMonochrome(input_data) {
  const processed_data = new Float32Array(input_data.length / 4);
  for (let x = 0; x < 28; x++) {
    for (let y = 0; y < 28; y++) {
      processed_data[x + y * 28] = input_data[4 * (x + y * 28)];
    }
  }
  return processed_data;
}

// Get class with highest probability
function getPredictionFromArray(prediction_array) {
  const max = prediction_array.reduce((current_max_value, item_value) => {
    return item_value > current_max_value ? item_value : current_max_value;
  }, 0);
  return prediction_array.indexOf(max);
}
