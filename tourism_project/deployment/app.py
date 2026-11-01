"""Streamlit Community Cloud entry point; only pre-contact customer information."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title='Visit with Us | Wellness', page_icon='🌿', layout='wide')
st.markdown("""<style>
.stApp {background: #f5f8f8;}
h1,h2,h3 {color:#134f54;}
div.stButton > button {background:#126d73;color:white;border:0;border-radius:10px;}
</style>""",unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # Load only the trusted model version committed by this repository's pipeline.
    return joblib.load(HERE/'model.joblib'),json.loads((HERE/'metadata.json').read_text())

try:
    bundle, metadata = load_assets()
except FileNotFoundError:
    st.error('The model is not available yet. Complete the training workflow first.')
    st.stop()

st.caption('VISIT WITH US  /  MARKETING INTELLIGENCE')
st.title('A more thoughtful way to find your next traveller')
st.write('Estimate purchase propensity before making contact. Use this score to prioritize outreach for the wellness campaign.')
st.info('This model learns historical package purchases. Wellness-specific performance needs validation in a new campaign.')

with st.form('customer_profile'):
    st.subheader('Customer profile')
    a,b,c=st.columns(3)
    with a:
        age=st.number_input('Age',min_value=18,max_value=100,value=37)
        city=st.selectbox('City tier',[1,2,3])
        occupation=st.selectbox('Occupation',['Salaried','Small Business','Large Business','Free Lancer'])
        designation=st.selectbox('Designation',['Executive','Manager','Senior Manager','AVP','VP'])
    with b:
        gender=st.selectbox('Gender',['Female','Male'])
        marital=st.selectbox('Marital status',['Single','Unmarried','Married','Divorced'])
        income=st.number_input('Monthly income (dataset currency)',min_value=0,value=23000,step=500)
        trips=st.number_input('Trips per year',min_value=0,max_value=50,value=3)
    with c:
        people=st.number_input('Total people travelling',min_value=1,max_value=20,value=3)
        children=st.number_input('Children below age 5 travelling',min_value=0,max_value=20,value=1)
        stars=st.selectbox('Preferred property stars',[3,4,5])
        passport=st.checkbox('Valid passport',value=True)
        car=st.checkbox('Owns a car',value=False)
    submitted=st.form_submit_button('Estimate purchase propensity',use_container_width=True)

if submitted:
    if children > people:
        st.error('The number of children cannot exceed the total people travelling.')
    else:
        # Match training schema explicitly; no pitch, follow-up, or satisfaction data.
        inputs = pd.DataFrame([{'Age':float(age),'CityTier':city,
            'NumberOfPersonVisiting':people,'PreferredPropertyStar':float(stars),
            'NumberOfTrips':float(trips),'Passport':int(passport),'OwnCar':int(car),
            'NumberOfChildrenVisiting':float(children),'MonthlyIncome':float(income),
            'Occupation':occupation,'Gender':gender,'MaritalStatus':marital,'Designation':designation}])
        inputs=inputs[bundle['features']]
        score=float(bundle['pipeline'].predict_proba(inputs)[0,1])
        priority=score>=bundle['threshold']
        left,right=st.columns(2)
        left.metric('Purchase propensity score',f'{score:.1%}')
        right.metric('Outreach priority','Higher priority' if priority else 'Standard priority')
        st.caption(f"Decision threshold: {bundle['threshold']:.3f}. Scores are model estimates, not calibrated guarantees.")
        if age>61 or trips>22 or people>5 or income<1000 or income>98678:
            st.warning('Some inputs are outside the observed training ranges. Treat this score with extra caution.')
        with st.expander('Review entered profile'):
            st.dataframe(inputs,hide_index=True)

with st.expander('Model quality and appropriate use'):
    m=metadata['test_metrics']
    st.write(f"Model: {metadata['algorithm']} · Holdout recall: {m['recall']:.1%} · Holdout precision: {m['precision']:.1%} · Average precision: {m['average_precision']:.3f}")
    st.write('The threshold maximizes training out-of-fold F2, favoring buyer recall. It is not an optimized profit threshold. Validate against contact costs and campaign outcomes before rollout.')
    st.write('Inputs are used for this prediction and are not written to a customer database. Gender and other attributes need fairness review before operational use. Do not use the score to deny travel services.')
st.caption('Pre-contact signals only • Human review • Monitor outcomes and data drift')
