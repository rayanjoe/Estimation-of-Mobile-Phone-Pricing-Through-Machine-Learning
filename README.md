# Estimation-of-Mobile-Phone-Pricing-Through-Machine-Learning

Project Overview
This project addresses the challenge of setting competitive prices in the mobile phone market. By analyzing features such as RAM, internal memory, and others, the project aims to classify mobile phones into different price ranges. This approach assists in devising effective pricing strategies for new mobile products.

Dataset
The dataset comprises 2000 entries with features like battery power, Bluetooth availability, clock speed, RAM, and others, culminating in a target variable price_range indicating the phone's cost category.

Objectives
Data Exploration and Preprocessing: Initial data examination and cleaning to prepare for modeling.
Model Building: Develop various classification models to predict the price range.
Prediction: Apply the models to unseen data to predict price ranges.
Models Implemented
Several machine learning models were explored, including:

Support Vector Machine (SVM)
Decision Tree (DT)
Random Forest (RF)
XGBOOST
CATBOOST
Artificial Neural Network (ANN)
Key Technologies and Libraries Used
Pandas and NumPy for data manipulation
Matplotlib and Seaborn for data visualization
Scikit-learn for machine learning models and evaluation
XGBoost and CatBoost for gradient boosting models
TensorFlow and Keras for neural network implementation
Model Training and Evaluation
Models were trained using a split of training, validation, and test sets to ensure a robust evaluation. Techniques like PCA (Principal Component Analysis) were employed for dimensionality reduction, improving model performance by focusing on the most informative features.

Results and Findings
The SVM classifier showed promising results, further enhanced by PCA, achieving an accuracy of 95.75% on the validation set.
Decision Trees and Random Forests provided insight into feature importance but were more prone to overfitting compared to ensemble methods like XGBOOST and CATBOOST.
XGBOOST and CATBOOST models were finely tuned for performance, reaching accuracies above 90%, showcasing the effectiveness of gradient boosting techniques in handling tabular data for classification.
An extensive hyperparameter tuning phase was conducted, particularly for models like SVM and CATBOOST, utilizing techniques like grid search to identify optimal settings.
Conclusion
The project successfully demonstrates the application of machine learning to predict mobile phone price ranges based on their specifications. The findings indicate that gradient boosting models, particularly XGBOOST and CATBOOST, offer superior performance in this domain. Future work could explore deeper neural network architectures and more sophisticated feature engineering to further enhance model accuracy.
