importScripts(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.2.7/dist/tf.min.js"
);
tf.setBackend("cpu");
const model = await tf.loadLayersModel("conv.json");
input = tf.tensor(seq);
predictOut = model.predict(input);
predictOut.dispose();
console.log(model);