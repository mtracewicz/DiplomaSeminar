function updateSelectFileLabel() {
  let fileName = selector.value;
  fileName = String(fileName).split("\\");
  fileName = fileName[fileName.length - 1];
  document.getElementById("inputLabel").innerHTML = fileName;
  document.getElementById("predict").disabled = false;
}

const selector = document.getElementById("inputGroupFile01");
selector.addEventListener("change", updateSelectFileLabel);

const btn = document.getElementById("predict");
btn.addEventListener("click", predict);

function predict() {
  var worker = new Worker("worker.js");
  worker.postMessage();
}
