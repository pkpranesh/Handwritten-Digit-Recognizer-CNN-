# Handwritten Digit Recognizer CNN

A convolutional neural network (CNN) that recognizes handwritten digits from the
MNIST dataset. The trained model is exposed through a simple Streamlit web app
where you can upload a digit image and view the prediction confidence for every
digit from 0 to 9.

## Features

- Trains a CNN on the TensorFlow MNIST dataset.
- Normalizes and reshapes images to the `28 x 28 x 1` format expected by the model.
- Saves the trained model to `model/mnist_cnn.h5`.
- Displays the predicted digit and confidence scores in a Streamlit interface.
- Automatically resizes uploaded PNG, JPG, or JPEG images to `28 x 28`.

## Requirements

- Python 3.9 or newer
- pip

The required Python packages are listed in
[requirements.txt](requirements.txt).

## Installation

1. Clone the repository and move into the project directory:

   ```bash
   git clone https://github.com/pkpranesh/Handwritten-Digit-Recognizer-CNN-.git
   cd Handwritten-Digit-Recognizer-CNN-
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the application

The repository includes a trained model at `model/mnist_cnn.h5`. Start the
Streamlit app with:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, upload an image containing one handwritten
digit, and the app will display the predicted digit, its confidence, and the
probabilities for all ten classes.

## Training the model

To retrain the CNN using the MNIST dataset, run:

```bash
python model_train.py
```

The training script downloads the MNIST dataset through TensorFlow, trains for
10 epochs, evaluates the model on the test set, saves the training curves to
`training_history.png`, and writes the model to `model/mnist_cnn.h5`.

Retraining replaces the existing model file and may take several minutes
depending on your hardware.

## Project structure

```text
.
├── app.py                  # Streamlit prediction interface
├── model_train.py          # MNIST data loading and CNN training
├── model/
│   └── mnist_cnn.h5       # Saved trained Keras model
├── training_history.png    # Training and validation plots
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Image preprocessing

Uploaded images are converted to grayscale, resized to `28 x 28` pixels, scaled
to values between 0 and 1, and reshaped before being passed to the CNN. For the
best results, upload a clear image containing a single centered digit with
contrast similar to the MNIST dataset.

## License

No license has been specified for this project.
