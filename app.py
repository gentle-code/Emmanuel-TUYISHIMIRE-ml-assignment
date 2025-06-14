import streamlit as st
import pandas as pd
import joblib

# Load the model (save it first from your notebook)
model = joblib.load('titanic_model.pkl')

st.title('Titanic Survival Predictor')

# Input widgets
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.slider('Number of Siblings/Spouses', 0, 8, 0)
parch = st.slider('Number of Parents/Children', 0, 6, 0)
fare = st.slider('Fare', 0, 600, 50)
embarked = st.selectbox('Port of Embarkation', ['Cherbourg', 'Queenstown', 'Southampton'])
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
title = st.selectbox('Title', ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'])

# Convert inputs to model format
sex = 1 if sex == 'Male' else 0
embarked_map = {'Cherbourg': 1, 'Queenstown': 2, 'Southampton': 0}
embarked = embarked_map[embarked]
title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
title = title_map[title]

# Predict button
if st.button('Predict Survival'):
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked, family_size, is_alone, title]],
                            columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title'])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader('Prediction:')
    if prediction == 1:
        st.success('Survived (Probability: {:.2f}%)'.format(probability * 100))
    else:
        st.error('Did Not Survive (Probability: {:.2f}%)'.format((1 - probability) * 100))