
# Resume Classification Project

This project focuses on classifying resumes into various categories using machine learning techniques. The application is built to process resume text data and predict the category of a given resume.

## Project Structure

- **`train_model.py`**: Script used to train the classification models. It reads the dataset, processes the text data, and trains various models including logistic regression and ensemble models.
- **`app/`**: 
  - **`app.py`**: A Python script that likely serves as an application or API to interact with the trained model. This script may allow users to input resume data and receive classification predictions.
- **`data/`**:
  - **`UpdatedResumeDataSet.csv`**: The dataset containing resume data used for training and testing the models.
- **`models/`**:
  - **`ensemble_model.pkl`**: The saved ensemble model for classifying resumes.
  - **`label_encoder.pkl`**: The label encoder used to convert categories into numerical labels during training.
  - **`logistic_regression_model.pkl`**: The saved logistic regression model for classification.
  - **`tfidf_vectorizer.pkl`**: The TF-IDF vectorizer used to transform resume text into numerical features for model input.

## How to Use

### Training the Model

To train the model, execute the `train_model.py` script. This script will process the dataset, train the models, and save them in the `models/` directory.

```bash
python train_model.py
```

### Running the Application

To run the application, use the `app.py` script located in the `app/` directory. This script likely starts a server or a command-line interface where you can input resumes and get predictions.

```bash
python app/app.py
```

### Dependencies

Ensure you have the necessary dependencies installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

(*Note*: A `requirements.txt` file should be included with the necessary Python packages listed, if not already present.)

## Dataset

The dataset used for training is included in the `data/` directory as `UpdatedResumeDataSet.csv`. This dataset contains resumes with corresponding labels that indicate the category to which each resume belongs.

## Models

The trained models are stored in the `models/` directory:
- **Ensemble Model**: `ensemble_model.pkl`
- **Logistic Regression Model**: `logistic_regression_model.pkl`
- **TF-IDF Vectorizer**: `tfidf_vectorizer.pkl`
- **Label Encoder**: `label_encoder.pkl`

These models are used by the application to predict the category of a given resume.

## License

Include your licensing information here, if applicable.
