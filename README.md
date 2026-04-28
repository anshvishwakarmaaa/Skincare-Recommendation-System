🧴 AI Skincare Recommendation System

An AI-powered skincare analysis and recommendation system that uses deep learning to analyze skin images and predict possible skin conditions. The project includes dataset processing, model training, evaluation, and a web interface for predictions.

🚀 Features

Skin image classification using deep learning

Automated dataset preprocessing pipeline

PyTorch model training and evaluation

Visualization of performance metrics

Streamlit web application for predictions

Upload skin images and get model predictions

🧠 Technologies Used

Python

PyTorch

NumPy

Pandas

OpenCV

Scikit-learn

Matplotlib

Streamlit

⚙️ Installation

Clone the repository:

git clone:
 https://github.com/anshvishwakarmaaaSkincare-Recommendation-System.git
cd Skincare-Recommendation-System

Install dependencies:

pip install -r requirements.txt

🏋️ Model Training

To train the model:

python src/pytorch_train.py

This will train the CNN model using the prepared dataset.

📊 Model Analysis

To analyze model performance:

python src/pytorch_analyzer.py

Outputs include:

Confusion matrix
Performance metrics
Model evaluation statistics

Results will be stored in:

model_analysis/
🌐 Run the Web Application

To start the Streamlit interface:

streamlit run streamlit_app.py

This will open a web interface where users can upload skin images and get predictions.

🔮 Future Improvements

Improve dataset size and diversity

Add more skin condition classes

Deploy the model online

Improve UI of the web application

Add explainable AI for predictions