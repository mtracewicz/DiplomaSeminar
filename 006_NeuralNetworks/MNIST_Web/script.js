// Creating new WebWorker
const worker = new Worker("worker.js");

// Display prediction received from WebWorker
worker.onmessage = (e) => {
  const label = document.getElementById("prediction");
  label.innerText = `Your number is: ${e.data}`;
};

/*
  Register handler which changes files label and
  will allow to predict only after file is selected
*/
const selector = document.getElementById("inputGroupFile01");
selector.addEventListener("change", updateSelectFile);

// Update selected image and allow predictions
function updateSelectFile() {
  updateSelectedLabel();
  loadImage();
  enablePrediction();
}

// Update label with selected file name
function updateSelectedLabel(){
  document.getElementById("inputLabel").innerHTML = getFileName();
}

// Parse file name to drop fake path
function getFileName() {
  let fileName = selector.value;
  fileName = String(fileName).split("\\");
  fileName = fileName[fileName.length - 1];
  return fileName;
}

// Load image uploaded by user
function loadImage(){
  const reader = new FileReader();
  reader.onload = (e) => {
    img_url = e.target.result;
    displayImage(img_url);
  };
  reader.readAsDataURL(getSelectedFile());
}

// Returns file selected by user
function getSelectedFile(){
  return document.getElementById("inputGroupFile01").files[0];
}

// Display image selected by user
function displayImage(img_url) {
  const img = document.getElementById("predictionImage");
  if (img === null) {
    createImageAndInsertItIntoDocument(img_url);
  } else {
    img.src = img_url;
  }
  resetLabel();
}

// Create image and insert it into document
function createImageAndInsertItIntoDocument(img_url) {
  const row = document.getElementById("mrow");
  const img = document.createElement("img");
  img.id = "predictionImage";
  img.src = img_url;
  row.insertBefore(img, row.firstChild);
}

// Resetting prediction label
function resetLabel(){
  const label = document.getElementById("prediction");
  label.innerText = "Your prediction will appear here."
}

// Enable predict button
function enablePrediction(){
  document.getElementById("predict").disabled = false;
}

// Register handler for clicking predict button
const btn = document.getElementById("predict");
btn.addEventListener("click", handlePredictionClick);

// Prepare data which will be sent to Tensorflow model, then post it as message
async function handlePredictionClick(){
  const context = document.createElement("canvas").getContext("2d");
  const image = await getPredictionImageData();
  context.drawImage(image, 0, 0);
  const data = context.getImageData(0, 0, image.width, image.height).data;
  worker.postMessage(data);
}

// Returns bitmap from rendered image
function getPredictionImageData() {
  const img = document.getElementById("predictionImage");
  return createImageBitmap(img);
}
