// Import Tensorflow JS
importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.2.7/dist/tf.min.js"
);

// Make prediction and send back answer
onmessage = async function predict(e) {
  tf.setBackend("cpu");
  const input = preprocessImage(e.data);
  const model = await tf.loadLayersModel("model/model.json");
  const predictionArray = await model.predict(input).data();
  postMessage(getPredictionFromArray(predictionArray));
};

// Prepare data to fit model input
function preprocessImage(image) {
  const scaledData = scaleData(image);
  const preprocessedData = makeMonochrome(scaledData);
  return tf.tensor(preprocessedData).asType("float32").reshape([1, 28, 28, 1]);
}

// Scale data from integers [0,255] to floats [0.0,1.0]
function scaleData(data) {
  return new Float32Array(data).map((i) => i / 255.0);
}

// Get monochrome version from RGBA
function makeMonochrome(inputData) {
  let processed_data = new Float32Array(inputData.length / 4);
  for (let x = 0; x < 28; x++) {
    for (let y = 0; y < 28; y++) {
      /* We can assign red color from input data to processed data,
      because as the pictures are monochrome in their nature values of 
      red,green,blue are always the same.
      */
      processed_data[x + y * 28] = inputData[4 * (x + y * 28)];
    }
  }
  return processed_data;
}

// Get class with highest probability
function getPredictionFromArray(predictionArray) {
  const max = predictionArray.reduce((current_max_value, item_value) => {
    return item_value > current_max_value ? item_value : current_max_value;
  }, 0);
  return predictionArray.indexOf(max);
}
