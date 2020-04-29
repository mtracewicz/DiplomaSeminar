importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.2.7/dist/tf.min.js"
);

onmessage = async function predict(e) {
  tf.setBackend("cpu");
  const input = preprocessImage(e.data);
  console.log(await input.data())
  const model = await tf.loadLayersModel("model/model.json");
  const predictionArray = await model.predict(input).data();
  postMessage(getPredictionFromArray(predictionArray));
};

function preprocessImage(image) {
  return tf.browser
    .fromPixels({ data: image, width: 28, height: 28 }, 1)
    .asType("float32")
    .reshape([1, 28, 28, 1]);
}

function getPredictionFromArray(predictionArray) {
  let max = 0;
  predictionArray.forEach((item) => {
    if (item > max) {
      max = item;
    }
  });
  return predictionArray.indexOf(max);
}
