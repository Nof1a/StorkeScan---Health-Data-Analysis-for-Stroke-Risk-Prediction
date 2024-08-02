# streamlit_app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import csv

# Load the trained model
model = joblib.load('rf_new.pkl')

# Define the app title
st.title('Stroke Prediction')

# User inputs
gender = st.selectbox('الجنس', ('ذكر', 'أنثى'))
gender = 0 if gender == 'ذكر' else 1

age = st.number_input('العمر', min_value=0, max_value=120, step=1, format='%d')

hypertension = st.selectbox('ارتفاع ضغط الدم', ('لا', 'نعم'))
hypertension = 0 if hypertension == 'لا' else 1

heart_disease = st.selectbox('أمراض القلب', ('لا', 'نعم'))
heart_disease = 0 if heart_disease == 'لا' else 1

ever_married = st.selectbox('هل تزوجت من قبل', ('لا', 'نعم'))
ever_married = 0 if ever_married == 'لا' else 1

work_type = st.selectbox('نوع العمل', ('خاص', 'عمل حر', 'وظيفة حكومية', 'أطفال', 'لم يعمل أبداً'))
work_type_dict = {'خاص': 0, 'عمل حر': 1, 'وظيفة حكومية': 2, 'أطفال': 3, 'لم يعمل أبداً': 4}
work_type = work_type_dict[work_type]

Residence_type = st.selectbox('نوع السكن', ('ريف', 'حضر'))
Residence_type = 0 if Residence_type == 'ريف' else 1

avg_glucose_level = st.number_input('متوسط مستوى الجلوكوز')

bmi = st.number_input('مؤشر كتلة الجسم')

smoking_status = st.selectbox('حالة التدخين', ('لم يدخن أبداً', 'دخن سابقاً', 'يدخن'))
smoking_status_dict = {'لم يدخن أبداً': 0, 'دخن سابقاً': 1, 'يدخن': 2}
smoking_status = smoking_status_dict[smoking_status]

# Predict button
if st.button('تنبؤ'):
    input_features = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])
    prediction = model.predict(input_features)
    
    # Display prediction result
    if prediction[0] == 0:
        st.write('نتائجك جيدة، استمر في العادات الصحية!')
    else:
        st.write('تبدو صحتك جيدة، لكن استمر في مراقبة صحتك!')
    
    # Prepare new data
    new_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status,
        'stroke': prediction[0]
    }

    # Button to update CSV file
    if st.button('تحديث ملف البيانات'):
        # Read dataset
        file_path = 'clean_data.csv'
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            data = pd.DataFrame(columns=new_data.keys())
        
        # Add new data to DataFrame
        data = data.append(new_data, ignore_index=True)
        
        # Save updated DataFrame to CSV
        data.to_csv(file_path, index=False)
        
        st.write('تم تحديث ملف البيانات بنجاح')

        # Append new row to CSV file directly
        new_row = [
            gender, age, hypertension, heart_disease, ever_married,
            work_type, Residence_type, avg_glucose_level, bmi, smoking_status, prediction[0]
        ]
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

        st.write('تم إضافة البيانات الجديدة إلى ملف CSV بنجاح')

        # Display new data
        st.write('البيانات الجديدة التي تمت إضافتها:')
        st.dataframe(data.tail(5))  # Display last 5 rows