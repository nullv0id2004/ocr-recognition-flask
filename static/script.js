const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const resultDiv = document.getElementById('result');


navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((err) => {
        console.error('Error accessing webcam: ', err);
    });

captureButton.addEventListener('click', async () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg');
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataUrl }),
        });
        const result = await response.json();
        if (result.prediction !== undefined) {
            resultDiv.textContent = `Predicted Character: ${result.prediction}`;
        } else {
            resultDiv.textContent = `Error: ${result.error}`;
        }
    } catch (error) {
        console.error('Error during prediction: ', error);
        resultDiv.textContent = 'An error occurred.';
    }
});
