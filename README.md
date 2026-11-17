# Indian Cattle and Buffalo Breed Recognition

A deep learning web application that identifies Indian cattle and buffalo breeds from a photograph and returns breed information in English, Hindi or Tamil.

Breed identification matters for livestock registration, breeding programmes and veterinary care, but it usually depends on an expert looking at the animal. This project puts a trained image classifier behind a simple upload page so a farmer or field worker can get an answer from a phone photo.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow_/_Keras-FF6F00?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## Supported breeds

| Type | Breeds |
|---|---|
| Cattle | Alambadi, Bargur, Gir, Kangayam, Pulikulam, Umblachery |
| Buffalo | Murrah, Toda, generic buffalo |

Most of the cattle breeds are native to Tamil Nadu. For each prediction the app also shows the breed's origin and key characteristics from a built-in breed database.

## How it works

1. **Upload:** the user uploads a PNG, JPEG or GIF image (up to 16 MB) through the web interface.
2. **Preprocess:** the image is resized to 224 x 224 and normalised (`utils/image_preprocessing.py`).
3. **Classify:** a convolutional neural network predicts one of nine classes with a confidence score.
4. **Explain:** the predicted breed is looked up in `utils/breed_database.py` and shown in the selected language.

## Model

- **Production model:** transfer learning on **EfficientNetB0** pre-trained on ImageNet. The base is frozen for the first training phase and then fine-tuned end to end (`models/model_trainer.py`).
- **Alternatives:** `generate_model.py` can also build a four-block CNN from scratch or a VGG16 transfer-learning model, which is useful for comparison or for running without a pre-trained backbone.
- **Training pipeline:** data augmentation, early stopping and checkpoint callbacks, plus evaluation with a confusion matrix and training-history plots.
- The trained weights are in `models/breed_recognition_model.h5`, and the label order is in `models/class_mapping.json`.

## Features

- Web interface built with Flask and Jinja templates
- English, Hindi and Tamil interface and breed descriptions (`translations/`)
- `/health` and `/model-info` endpoints for monitoring
- Clear errors for unsupported file types and oversized uploads

## Getting started

```bash
git clone https://github.com/mithileshofficial06/CNN_Breed_Classifier.git
cd CNN_Breed_Classifier
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000.

### Training your own model

Place images in `data/raw/<type>/<breed>/` and run:

```bash
python models/model_trainer.py
```

## Project structure

```
app.py                    Flask app: upload, prediction, language switching, health checks
config.py                 Paths, input size, upload limits, training settings
generate_model.py         Builds a baseline CNN or a VGG16 transfer-learning model
models/
  model_trainer.py        EfficientNetB0 training, fine-tuning and evaluation
  model_predictor.py      Loads the model and runs inference
  class_mapping.json      Class index to breed name
utils/
  image_preprocessing.py  Resizing, normalisation and augmentation
  breed_database.py       Breed origin and characteristics
  language_support.py     Translation helpers
templates/, static/       Web interface
translations/             en, hi and ta strings
```
