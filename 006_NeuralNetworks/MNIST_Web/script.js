// Creating new WebWorker
const worker = new Worker("worker.js");

// Returns file selected by user
const getSelectedFile = () =>
  document.getElementById("inputGroupFile01").files[0];

// Allow prediction after file is selected, change file label
const selector = document.getElementById("inputGroupFile01");
selector.addEventListener("change", updateSelectFile);


// Update selected image and allow predictions
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

// Display image selected by user
function displayImage(img_url) {
  const img = document.getElementById("predictionImage");
  if (img === null) {
    createImage(img_url);
  } else {
    img.src = img_url;
  }
  resetLabel();
}

// Resetting prediction label
function resetLabel(){
  const label = document.getElementById("prediction");
  label.innerText = "Your prediction will appear here."
}

// Returns bitmap from rendered image
function getPredictionImageData() {
  const img = document.getElementById("predictionImage");
  return createImageBitmap(img);
}

// Create image and insert it into document
function createImage(img_url) {
  const row = document.getElementById("mrow");
  const img = document.createElement("img");
  img.id = "predictionImage";
  img.src = img_url;
  row.insertBefore(img, row.firstChild);
}

// Parse file name to drop fake path
function getFileName() {
  let fileName = selector.value;
  fileName = String(fileName).split("\\");
  fileName = fileName[fileName.length - 1];
  return fileName;
}

// Prepare data which will be sent to Tensorflow model, then post it as message
const btn = document.getElementById("predict");
btn.addEventListener("click", async () => {
  const context = document.createElement("canvas").getContext("2d");
  const image = await getPredictionImageData();
  context.drawImage(image, 0, 0);
  const data = context.getImageData(0, 0, image.width, image.height).data;
  worker.postMessage(data);
});

// Display prediction received from WebWorker
worker.onmessage = function displayPrediction(e) {
  const label = document.getElementById("prediction");
  label.innerText = `Your number is: ${e.data}`;
};
