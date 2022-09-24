# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Biological signatures and prediction of an immunosuppressive status—persistent critical illness—using machine learning methods")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters according to actual conditions")

firstalbuminchemistry = st.sidebar.slider("Albumin (g/dL)", 1.00, 6.00)
totalcalciumchemistry = st.sidebar.slider("Total serum calcium (mg/dL)", 3.00, 13.00)
RDWHematology = st.sidebar.slider("Red cell volume distributing width (RDW, %)", 8.00, 26.00)
pHBloodgas = st.sidebar.slider("Blood pH", 6.60, 7.70)
heartrateBPM = st.sidebar.slider("Heart rate (Beats per minute)", 26, 166)
SOFA = st.sidebar.slider("SOFA", 0, 20)
respiratoryfailure = st.sidebar.selectbox("Respiratory failure", ("No", "Yes"))
PneumoniaPneumonitis = st.sidebar.selectbox("Pneumonia", ("No", "Yes"))

if st.button("Submit"):
    rf_clf = jl.load("rf_clf_final_round.pkl")
    x = pd.DataFrame([[firstalbuminchemistry, totalcalciumchemistry, RDWHematology, pHBloodgas,
                              heartrateBPM, SOFA, respiratoryfailure, PneumoniaPneumonitis]],
                     columns=["firstalbuminchemistry", "totalcalciumchemistry", "RDWHematology", "pHBloodgas",
                              "heartrateBPM", "SOFA", "respiratoryfailure", "PneumoniaPneumonitis"])
    x = x.replace(["No", "Yes"], [0, 1])
    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of experiencing medical disputes: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.184:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.184:
        st.markdown(f"For patients in the low-risk group: Symptomatic treatment, such as fluid resuscitation, vasopressors, and management of pain, agitation, and delirium. The post-discharge recommendations included (1) identification of functional disability and impairments in swallowing and mental health, (2) review and adjust long-term medications to avoid medication errors, and (3) prevention of common causes of health deterioration, such as infection, acute heart and/or renal failure, and respiratory diseases.")
    else:
        st.markdown(f"For patients in the high-risk group: More attention is needed to be paid to improve nutritional status and immunity, maintain electrolyte and acid-base balance, curb infection, and promote respiratory recovery. If infection does not exist, preventive use of antibiotics is recommended. Modulation of endocrine responses to stress and early mobilization such as exercise and resistance training are also needed. Notably, dynamic monitor the changes of lymphocytes count may do lots of help to evaluate the developing status of PerCI.")

st.subheader('Introduction of the online application')
st.markdown('The online application was developed based on the highest performing model, the random forest. The model had an AUROC of 0.823 (95% CI: 0.757-0.889), Youden index of 1.571, and Brier score of 0.107. External validation also showed the AUROC could be up to 0.800 (95% CI: 0.688-0.912), Youden index was 1.637, and accuracy was up to 0.867. As expected, the prediction performance of the model also outperformed SOFA (AUROC: 0.631), OASIS (AUROC: 0.659), and SAPS II (AUROC: 0.578). ')

st.subheader('Risk stratification system')
st.markdown('Patients were stratified into two risk groups with regard to the threshold of the optimal model: Patients with an anticipated risk probability of less than 18.4% were categorized into the low-risk group, whereas patients with an anticipated risk probability of 18.4% or above were categorized into the high-risk group. Patients who were classified into the high-risk group were slightly above 10-time more vulnerable to develop PerCI than those in the low-risk group in the internal validation cohort (P<0.001). Similar results were observed in the external validation cohort and the number was more than 3 times (P=0.001).')
