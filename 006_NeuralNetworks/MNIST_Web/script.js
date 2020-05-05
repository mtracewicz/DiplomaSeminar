// Creating new WebWorker
const worker = new Worker("worker.js");

const getSelectedFile = () =>
  document.getElementById("inputGroupFile01").files[0];

// Allow prediction after file is selected, change file label
const selector = document.getElementById("inputGroupFile01");
selector.addEventListener("change", updateSelectFile);

function updateSelectFile() {
  const fileName = getFileName();
  document.getElementById("inputLabel").innerHTML = fileName;
  const reader = new FileReader();
  reader.onload = (e) => {
    img_url = e.target.result;
    displayImage(img_url);
  };
  reader.readAsDataURL(getSelectedFile());
  document.getElementById("predict").disabled = false;
}

function displayImage(img_url) {
  const img = document.getElementById("predictionImage");
  if (img === null) {
    createImage(img_url);
  } else {
    img.src = img_url;
  }
}

function getPredictionImageData() {
  const img = document.getElementById("predictionImage");
  return createImageBitmap(img);
}

function createImage(img_url) {
  const row = document.getElementById("mrow");
  const img = document.createElement("img");
  img.id = "predictionImage";
  img.src = img_url;
  row.insertBefore(img, row.firstChild);
}

function getFileName() {
  let fileName = selector.value;
  fileName = String(fileName).split("\\");
  fileName = fileName[fileName.length - 1];
  return fileName;
}

const btn = document.getElementById("predict");
btn.addEventListener("click", () => {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  getPredictionImageData().then((value) => {
    const image = value;
    ctx.drawImage(image, 0, 0);
    let img_data = ctx.getImageData(0, 0, image.width, image.height).data;
    worker.postMessage(img_data);
  });
});

// Display prediction received from WebWorker
worker.onmessage = function displayPrediction(e) {
  const label = document.getElementById("prediction");
  label.innerText = `Your number is: ${e.data}`;
};
