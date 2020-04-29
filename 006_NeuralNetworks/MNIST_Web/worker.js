importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.2.7/dist/tf.min.js"
);

onmessage = async function predict(e) {
  tf.setBackend("cpu");
  const input = preprocessImage(e.data)
  const model = await tf.loadLayersModel("model.json");
  predictOut = model.predict(input);
  this.postMessage(predictOut);
  predictOut.dispose();
};

function preprocessImage(image) {
   return tf.tensor(image)
    // .reshape([28,28,1])
    // .resizeNearestNeighbor([28, 28])
    .toFloat();
}