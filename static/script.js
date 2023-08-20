var uploadForm = document.getElementById('upload-form');
var uploadedImage = document.getElementById('uploaded-image');
const resultSection = document.getElementById('result');

uploadForm.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const imageUrl = URL.createObjectURL(file);
        uploadedImage.src = imageUrl;
    }
});
var i = 0;

uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent default form submission

    const formData = new FormData(uploadForm); // Create a FormData object

    try {
        const response = await fetch('/result', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            // Process the result if needed
            resultSection.textContent = `Ratinopathy Level: ${result.size} `;
            i += 1;
        } else {
            console.error('Image upload failed:', response.statusText);
        }
    } catch (error) {
        console.error('Error sending image:', error);
    }
})