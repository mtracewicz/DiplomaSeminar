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
  const reader = new FileReader();
  reader.onload = (e) => {
    let img_buff = e.target.result;
    img_buff = new Uint8Array(img_buff)
    worker.postMessage(img_buff);
  };
  reader.readAsArrayBuffer(getSelectedFile());
});

// Display prediction received from WebWorker
onmessage = function displayPrediction(e) {
  const label = document.getElementById("prediction");
  label.innerText = e.data;
};
