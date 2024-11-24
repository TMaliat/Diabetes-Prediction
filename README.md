Diabetes-Prediction
A ML based approach to early detect Type-2 Diabetes.

In this project, I implemented a Diabetes Prediction System using the PIMA Indians Diabetes Dataset, a widely recognized dataset for predicting Type 2 diabetes risk. The system utilizes key health parameters like glucose levels, blood pressure, BMI, age, and others to assess diabetes risk and provide personalized healthcare recommendations.

The PIMA Indians Diabetes Dataset contains 768 patient records from women of Pima Indian heritage, with eight features used for predicting Type 2 diabetes:

Pregnancies: Number of pregnancies the individual has had. 
Glucose: Blood glucose concentration. 
Blood pressure: Diastolic blood pressure (mm Hg). 
Skin thickness: Triceps skinfold thickness (mm). 
Insulin: 2-hour serum insulin (mu U/ml). 
BMI: Body mass index (kg/m²). 
Diabetes pedigree function: Genetic relationship to diabetes. 
Age: Age of the individual (years). This dataset helps assess Type 2 diabetes risk, enabling early detection and preventive care.

How It Works: The system uses machine learning algorithms to predict diabetes risk based on user inputs. Users submit health data via a React front-end, and a Flask backend processes it to generate predictions, classifying users as either diabetic or non-diabetic.

Tech Stack: Frontend: React.js (UI development) Backend: Flask (API handling & prediction processing) ML Model: Logistic Regression (prediction) Styling: Custom CSS (user interface design)

This project can be turned into an API for use in the healthcare sector:

Healthcare Providers: For early diabetes detection. Telemedicine: Remote health assessments. Health Apps: Personalized tracking of diabetes risk.

This Diabetes Prediction API can revolutionize diabetes care by enabling early diagnosis and personalized healthcare solutions. With further improvements in accuracy and security, it could be implemented in clinics, hospitals, and telemedicine platforms, making healthcare more accessible and effective.
