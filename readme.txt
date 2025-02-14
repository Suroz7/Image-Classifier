# Image Classification with ResNet-50

This project is a simple image classification tool using a pre-trained ResNet-50 model from Hugging Face. The application allows users to upload an image and get a predicted class label.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <project_directory>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Backend Inference

To run image classification using the model directly:
```bash
python src/infer.py
```

### Running the UI

To launch the interactive UI:
```bash
streamlit run src/app.py
```

This will open a browser where you can upload images and view predictions.

## Project Structure
```
project_directory/
│── src/
│   ├── infer.py  # Backend script for inference
│   ├── app.py    # Streamlit UI
│── data/         # Place your test images here
│── requirements.txt  # Dependencies
│── README.md  # Project instructions
```

## Notes
- Ensure you have a stable internet connection to fetch the model and class labels.
- Works best with images from ImageNet dataset.

---

Enjoy classifying images! 🚀

