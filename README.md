# Classification Neural Network Deployment on Cerebrium

This repository contains the codebase for deploying a pre-trained image classification neural network (PyTorch ResNet-18) on Cerebrium's serverless GPU platform. The model is first converted to ONNX format for efficient inference.

## Table of Contents

1.  [Features / Deliverables](#features--deliverables)
2.  [Project Structure](#project-structure)
3.  [Setup Instructions](#setup-instructions)
4.  [Model Details](#model-details)
5.  [Usage](#usage)
    * [1. Convert PyTorch Model to ONNX](#1-convert-pytorch-model-to-onnx)
    * [2. Run Local Tests (`test.py`)](#2-run-local-tests-testpy)
    * [3. Run FastAPI Application Locally](#3-run-fastapi-application-locally)
    * [4. Test Local FastAPI Application](#4-test-local-fastapi-application)
    * [5. Deploy to Cerebrium](#5-deploy-to-cerebrium)
    * [6. Test Deployed Model (`test_server.py`)](#6-test-deployed-model-test_serverpy)
6.  [Evaluation Criteria](#evaluation-criteria)
7.  [Future Improvements](#future-improvements)

---

## Features / Deliverables

This project includes the following key deliverables:

* `convert_to_onnx.py`: Script to convert the PyTorch model (`pytorch_model.py`) to the ONNX format (`mtailor_model.onnx`).
* `model.py`: Contains utility functions for ONNX model loading, prediction calls, and image preprocessing.
* `test.py`: Comprehensive test suite for local verification of image preprocessing, ONNX model loading, and end-to-end inference using sample images.
* `Dockerfile`: Custom Docker image definition for deploying the FastAPI application to Cerebrium.
* `app.py`: FastAPI application that serves the ONNX model via a `/predict` API endpoint, handling image preprocessing and inference requests.
* `test_server.py`: (Placeholder for) Codebase to make calls to the model deployed on Cerebrium, allowing testing of the deployed model and platform-specific tests.
* `README.md`: This comprehensive documentation file.

## Project Structure
cerebrium-image-classification/
├── Dockerfile                  # Defines the Docker image for deployment
├── requirements.txt            # Python dependencies
├── app.py                      # FastAPI application for model serving
├── mtailor_model/
│   ├── model/
│   │   ├── model.py            # ONNX model utility functions
│   │   ├── pytorch_model.py    # Original PyTorch model definition (provided)
│   │   ├── mtailor_model.onnx  # Converted ONNX model (generated)
│   │   └── pytorch_model_weights.pth # PyTorch model weights (downloaded)
│   └── src/
│       ├── n01440764_tench.jpeg    # Sample test image (Class ID 0)
│       └── n01667114_mud_turtle.jpeg # Sample test image (Class ID 35)
├── convert_to_onnx.py          # Script for ONNX conversion
├── test.py                     # Local test suite
├── test_server.py              # Script to test the deployed model (to be implemented)
└── README.md                   # This documentation file


## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <your-github-repo-url>
    cd cerebrium-image-classification
    ```
2.  **Create a Python Virtual Environment**:
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```
3.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download PyTorch Model Weights**:
    Download the `pytorch_model_weights.pth` file from the provided link and place it in the `mtailor_model/model/` directory.
    [https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0](https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0)
5.  **Place Sample Images**:
    Ensure `n01440764_tench.jpeg` and `n01667114_mud_turtle.jpeg` are in the `mtailor_model/src/` directory.

---

## Model Details

* **Type**: Image Classification Neural Network (ResNet-18 architecture).
* **Framework**: Originally PyTorch, converted to ONNX for deployment.
* **Dataset**: Trained on the ImageNet Dataset.
* **Input**: Image of size 224x224 pixels, RGB format.
* **Output**: An array of 1000 probabilities, one for each ImageNet class.
* **Performance Expectation**: Answers within 2-3 seconds in production.

**Preprocessing Steps (implemented in `model.py` and `pytorch_model.py`):**
Images must undergo specific preprocessing before being passed to the model:
1.  Convert to RGB format.
2.  Resize to 224x224 pixels (using bilinear interpolation).
3.  Divide pixel values by 255 (normalize to 0-1 range).
4.  Normalize using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]` per channel.

---

## Usage

### 1. Convert PyTorch Model to ONNX

This step generates `mtailor_model.onnx` from the PyTorch model and its weights.

```bash
python convert_to_onnx.py
