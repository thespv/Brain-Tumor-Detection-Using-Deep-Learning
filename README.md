# Brain Tumor Detection System

A deep learning-powered web application for detecting brain tumors from MRI scans. This system uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories: Glioma, Meningioma, Pituitary, and No Tumor.

## Features

- Upload MRI brain scan images for analysis
- Automatic detection of tumor presence and type
- Real-time prediction with confidence score
- User-friendly web interface
- Supports 4 classification categories: Glioma, Meningioma, Pituitary, No Tumor

## Tech Stack

- **Backend**: Flask (Python web framework)
- **Deep Learning**: TensorFlow/Keras
- **Frontend**: HTML, Bootstrap 5
- **Model**: CNN (Convolutional Neural Network)

## Project Structure

```
Brain-Tumor-Detection-System/
├── main.py                 # Flask application
├── models/
│   └── model.h5           # Trained CNN model
├── templates/
│   └── index.html         # Frontend template
├── uploads/               # Uploaded images directory
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── venv/                  # Virtual environment
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thespv/Brain-Tumor-Detection-Using-Deep-Learning
cd "Brain Tumor Detection System"
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python main.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

3. Upload an MRI brain scan image and click "Upload and Detect"

## Model Details

- **Input Size**: 128x128 pixels
- **Preprocessing**: Image normalized to [0,1] range
- **Output Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Confidence Score**: Probability percentage of the predicted class

## Requirements

- Python 3.8+
- TensorFlow 2.16.1
- Flask 3.0.0
- NumPy 1.24.3
- Pillow 10.1.0
- h5py 3.10.0

## License

This project is for educational and research purposes.
