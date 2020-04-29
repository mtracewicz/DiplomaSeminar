importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.2.7/dist/tf.min.js"
);

onmessage = async function predict(e) {
  tf.setBackend("cpu");
  const input = preprocessImage(e.data);
  const model = await tf.loadLayersModel("model/model.json");
  predictOut = model.predict(input);
  let predictionArray = await predictOut.data();
  let prediction = 0,
    index = 0,
    max = 0;
  predictionArray.forEach((i) => {
    if (i > max) {
      max = i;
      prediction = index;
    }
    index++;
  });
  this.console.log(predictionArray)
  postMessage(prediction);
  predictOut.dispose();
};

function preprocessImage(image) {
  let image2 = new Uint8Array(28 * 28);
  image2.set(image);
  return tf.tensor(image2).reshape([1, 28, 28, 1]).toFloat();
}
