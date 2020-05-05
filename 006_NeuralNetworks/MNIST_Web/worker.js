importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.2.7/dist/tf.min.js"
);

onmessage = async function predict(e) {
  tf.setBackend("cpu");
  const input = preprocessImage(e.data);
  const model = await tf.loadLayersModel("model/model.json");
  const predictionArray = await model.predict(input).data();
  postMessage(getPredictionFromArray(predictionArray));
};

function preprocessImage(image) {
  let data = new Float32Array(image);
  data = data.map(i => i/255.0);
  let processed_data = new Float32Array(data.length/4);
  for(let i = 0;i<processed_data.length;i++){
    if(data[4*i] != 0 || data[1+4*i] != 0 || data[2+4*i] != 0){
      processed_data[i] = 1;
    }
  }
  return tf.tensor(processed_data)
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
