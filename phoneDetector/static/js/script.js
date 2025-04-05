document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file-upload");
    const fileNameDisplay = document.getElementById("file-name");
    const form = document.querySelector("form");
    const submitButton = document.getElementById("submit-btn");
    const loader = document.getElementById("loader");

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
        } else {
            fileNameDisplay.textContent = "No file chosen";
        }
    });

    form.addEventListener("submit", function () {
        submitButton.style.display = "none";
        loader.style.display = "block";
    });
});
