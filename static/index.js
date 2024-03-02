document.getElementById("add-file-button-icon").addEventListener("click", function() {
  document.getElementById("add-file-button").click();
});

document.getElementById("add-file-button").addEventListener("change", function(event) {
  var files = event.target.files;
  var fileContainer = document.getElementById("file-container");

  for (var i = 0; i < files.length; i++) {
    var file = files[i];

    // Create a new list item for each file and append it to the file container
    var listItem = document.createElement("li");
    listItem.textContent = file.name + ", \u00A0";
    fileContainer.appendChild(listItem);
  }

  // Make the file container visible after a file is uploaded
  fileContainer.style.display = "flex";
});

document.getElementById("user-input").addEventListener("keypress", function(event) {
  if (event.key === 'Enter') {
    document.getElementById("submit-button").click();
  }
});

document.getElementById("submit-button").addEventListener("click", function() {
  var userInput = document.getElementById("user-input").value;
  var chatContainer = document.getElementById("chat-container");

  // creating a new card for each query and append it to the chat container
  var card = document.createElement("div");
  card.className = "card";

  var query = document.createElement("p");
  query.textContent = "User: " + userInput;
  query.className = "query";
  card.appendChild(query);

  // Send a POST request to the Flask server
  fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({userInput: userInput})
  })
  .then(response => response.json())
  .then(data => {
    var response = document.createElement("p");
    response.textContent = "-> " + data.result;
    response.className = "response";
    card.appendChild(response);

    chatContainer.appendChild(card);
  });

  // clear the input field
  document.getElementById("user-input").value = "";
});