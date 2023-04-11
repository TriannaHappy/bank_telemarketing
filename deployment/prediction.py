import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json

#Read the models and columns
with open('imputer_num.pkl', 'rb') as file_1:
    imputer_num=pickle.load(file_1)
    
with open('imputer_cat1.pkl', 'rb') as file_2:
    imputer_cat1=pickle.load(file_2)

with open('imputer_cat2.pkl', 'rb') as file_3:
    imputer_cat2=pickle.load(file_3)
    
with open('windsoriser.pkl', 'rb') as file_4:
    windsoriser=pickle.load(file_4)
    
with open('Stdscaler.pkl', 'rb') as file_8:
    Stdscaler=pickle.load(file_8)
    
with open('MMscaler.pkl', 'rb') as file_8:
    MMscaler=pickle.load(file_8)

with open('RBscaler.pkl', 'rb') as file_8:
    RBscaler=pickle.load(file_8)

with open('ohe_pipeline.pkl', 'rb') as file_8:
    ohe_pipeline=pickle.load(file_8)
    
with open('ode_pipeline.pkl', 'rb') as file_8:
    ode_pipeline=pickle.load(file_8)

with open('preprocessor.pkl', 'rb') as file_9:
    preprocessor=pickle.load(file_9)  
    
with open('logreg_pipe.pkl', 'rb') as file_9:
    logreg_pipe=pickle.load(file_9)  
    

with open('num_cols.txt', 'r') as file_5:
    num_cols=json.load(file_5)

with open('nom_cat_cols.txt','r') as file_6: 
    nom_cat_cols=json.load(file_6)
    
with open('ord_cat_cols.txt', 'r') as file_7:
    ord_cat_cols=json.load(file_7) 
    
with open('num_cols_norm.txt', 'r') as file_5:
    num_cols_norm=json.load(file_5)

with open('num_cols_skew.txt','r') as file_6: 
    num_cols_skew=json.load(file_6)
    
with open('num_cols_rob.txt', 'r') as file_7:
    num_cols_rob=json.load(file_7)
    

def run():
    with st.form(key='form_heart_failure'):
        age = st.number_input('age', min_value=10, max_value=120, value=50, help='age of Client')
        job = st.text_input('job', help='job of the client')
        marital = st.selectbox('marital', ('divorced', 'married', 'single'), index=1, help='marital status')
        education = st.selectbox('education', ('tertiary', 'secondary', 'primary'), index=1, help='education level')
        default = st.selectbox('default', ('no', 'yes'), index=1, help='if the client default the payment')
        balance = st.number_input('balance', min_value=-9000.0, max_value=110000.0, value=500.0, help='Balance of The Client in Euro')
        housing = st.selectbox('housing', ('no', 'yes'), index=1, help='if the client has a loan house')
        loan = st.selectbox('loan', ('no', 'yes'), index=1, help='if the client has a loan')
        
        
        st.markdown('---')
        contact = st.selectbox('contact', ('cellular', 'telephone'), index=1, help='contact comunication type')
        day = st.number_input('day', min_value=0, max_value=31, value=16, help='last contact day of the month')
        month = st.selectbox('month', ('jan', 'feb', 'mar','apr',
                                       'may','jun','jul','aug',
                                       'sep','oct','nov','dec'), index=0, help='last contact month of year')
        duration = st.number_input('duration', min_value=0, max_value=5500, value=180, help=' last contact duration, in seconds (numeric)')

        st.markdown('---')
        campaign = st.number_input('campaign', min_value=1, max_value=90, value=2, help='number of contacts performed during this campaign and for this client')
        pdays = st.number_input('pdays', min_value=-1, max_value=70, value=10, 
                                help='number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)')
        previous = st.number_input('previous', min_value=0, max_value=300, value=10, help='number of contacts performed before this campaign and for this client')
        poutcome = st.selectbox('poutcome', ('failure', 'other','success'), index=1, help='outcome of the previous marketing campaign')
        st.markdown('---')
        submitted = st.form_submit_button('Predict')

    data_inf = {
        'age' : age,
        'job' : job, 
        'marital' :  marital,
        'education' : education,
        'default' : default,
        'balance' : balance,
        'housing':housing,
        'loan':loan,
        'contact':contact,
        'day':day,
        'month' : month,
        'duration' : duration,
        'campaign' : campaign,
        'pdays' : pdays,
        'previous' : previous,
        'poutcome' : poutcome

    }
    df=pd.DataFrame([data_inf])
    st.dataframe(df)

    if submitted:
        # categorize the education by the level of education
        job_cat=[]
        for i in range (df.shape[0]):    
            if ((df['job'][i] == "blue-collar") | (df['job'][i] == "technician")):
                job_cat.append('blue_collar')
            elif ((df['job'][i] == "management") | (df['job'][i] == "admin.")):
                job_cat.append('white_collar')
            elif ((df['job'][i] == 'unemployed')| (df['job'][i] == 'student')):
                job_cat.append("unemployed")
            else:
                job_cat.append(df['job'][i])
        df['job_cat']=job_cat
        
        # Labelling the month into month_num
        df['month_num'] = df['month'].apply(lambda x: datetime.strptime(x, '%b').month)

        # categorize the education by the level of education
        active_month=[]
        for i in range (df.shape[0]):    
            if (df['month_num'][i] in [1,2,4,6,7,8]):
                active_month.append('active')
            elif df['month_num'][i]==5:
                active_month.append('most_active')
            else:
                active_month.append('less_active')
        df['active_month']=active_month
        
        # Feature Selection
        X_test = df.drop(['job', 'month', 'contact', 'poutcome'], axis=1)

        # Split the X based on the column type
        X_test_num = X_test[num_cols]
        X_test_cat_nom = X_test[nom_cat_cols]
        X_test_cat_ord = X_test[ord_cat_cols]

        # Impute missing value in test data
        X_test_num=pd.DataFrame(imputer_num.transform(X_test_num), columns=X_test_num.columns.values)
        X_test_cat_nom=pd.DataFrame(imputer_cat1.transform(X_test_cat_nom), columns=X_test_cat_nom.columns.values)
        X_test_cat_ord=pd.DataFrame(imputer_cat2.transform(X_test_cat_ord), columns=X_test_cat_ord.columns.values)

        # Handling Outlier
        X_test_num_capped = windsoriser.transform(X_test_num)

        # Define the numerical columns
        X_test_num_norm = X_test_num_capped[num_cols_norm]
        X_test_num_skew = X_test_num_capped[num_cols_skew]
        X_test_num_rob = X_test_num_capped[num_cols_rob]

        #Transform the numerical test data
        X_test_num_norm_scaled = pd.DataFrame(Stdscaler.transform(X_test_num_norm), columns=num_cols_norm)
        X_test_num_skew_scaled = pd.DataFrame(MMscaler.transform(X_test_num_skew), columns=num_cols_skew)
        X_test_num_rob_scaled = pd.DataFrame(RBscaler.transform(X_test_num_rob), columns=num_cols_rob)

        # Concate the imbalance X data
        X_test_imbalanced=pd.concat([X_test_num_norm_scaled,X_test_num_skew_scaled,X_test_num_rob_scaled,
                                    X_test_cat_nom, X_test_cat_ord], axis=1)
        
        # Predict the target
        y_pred_test_logreg_pipe = logreg_pipe.predict(X_test_imbalanced)
        result_client = 'yes' if y_pred_test_logreg_pipe==1 else 'no'



        st.write('# Does this client subsribed the term deposit? ', result_client)
        
if __name__ == '__main__':
    run()