# Grad School Admission Chance Predictor

This app has been built using Streamlit and can be deployed with Streamlit community cloud. It predicts graduate school admission chances based on academic profile inputs.

## Features
- User-friendly interface powered by Streamlit
- Interactive sliders for inputting academic metrics
- Real-time prediction of admission probability
- Model trained on historical admission data

## Dataset
The application is trained on graduate admissions data including features like:
- GRE Score (260-340)
- TOEFL Score (80-120)
- University Rating (1-5)
- Statement of Purpose (SOP) Strength (1.0-5.0)
- Letter of Recommendation (LOR) Strength (1.0-5.0)
- Undergraduate GPA (6.0-10.0)
- Research Experience (Yes/No)

## Technologies Used
- **Streamlit**: For building the web application
- **Scikit-learn**: For model training and evaluation
- **Pandas** and **NumPy**: For data preprocessing and manipulation

## Model
The predictive model is trained using graduate admissions data. It includes:
- Data preprocessing (scaling numerical features)
- Machine learning model (likely Logistic Regression or similar)
- Probability prediction for admission chances

## How to Use
1. Adjust the sliders in the sidebar to input student profile
2. Click "Predict Admission Chance" button
3. View the predicted probability (0-1 scale)

## Installation (for local deployment)
To run the application locally:

1. Clone the repository:
   ```bash
git clone https://github.com/riyaroy-python/Neural_networks.git 


2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

## Future Enhancements
- Add visualizations of prediction trends
- Include feature importance explanations
- Support for multiple prediction models
- Comparison with historical admission data

## Acknowledgments
Dataset sourced from [Kaggle Graduate Admissions Dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)

Thank you for using the Grad School Admission Chance Predictor! Feel free to share your feedback.
