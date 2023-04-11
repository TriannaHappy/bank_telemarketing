import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
from PIL import Image

# Untuk melebarkan streamlit, harus diletakkan setelh import
# Ketika dieksekusi akan mempengaruhi main dan prediction
# Tidak perlu dijalankan dalam fungsi
st.set_page_config(
    page_title='Bank Telemarketing',
    layout='wide',
    initial_sidebar_state='expanded'
)


# bagian bawah ini tidak bisa dijalankan jika tidak dieksekusi
def run():
    #Membuat Title
    st.title('Bank Telemarketing with Machine Learning Classifier Model')

    # Membuat Sub Header
    st.subheader('EDA for Bank Client Dataset')

    # Menambahkan Gambar
    image=Image.open('image_bank.jpg')
    st.markdown(
    """
    <style>
    img {
        cursor: pointer;
        transition: all .2s ease-in-out;
    }
    img:hover {
        transform: scale(1.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.image(image, caption='Bank Telemarketing')

    # Menambah Deskripsi
    st.write('Made by *Happy Trianna*')
    
    # Membuat Garis Lurus
    st.markdown('---')

    # Magic Syntax
    '''
    Pada page kali ini, pennulis akan melakukan eksplorasi sederhana.
    Dataset yang digunakan adalah dasaet Bank Client dari Bank Portugal.
    Dataset ini berasal dari web archive.ics.uci.edu.
    '''

    # Show DataFrame
    st.write('#### Head of Bank Marketing Dataset')
    df = pd.read_csv('bank-full.csv', sep=';')
    st.dataframe(df.head(10))

    st.write('#### Describe of The Bank Marketing Dataset')
    st.dataframe(df.describe())

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

    st.markdown('---')

    # Plot of the correlation
    # Impute target with binary value of y column
    st.write('#### Correlation of The Features To The Target')
    df['target']=df['y'].apply(lambda x: 1 if x=='yes' else 0)
    corr = df.corr()
    fig=px.bar(corr['target'][:-1].sort_values())
    st.plotly_chart(fig)

    # Plot of the month where operator contacts the clients
    st.write("#### Number of Contacts Operators Make Each Month")
    fig = plt.figure(figsize=(4,4))
    sns.countplot(x=df['month_num'])
    st.pyplot(fig)

    #Check the distribution of clients, categorized by target who subsribed the term deposit
    st.write('#### Clients Subscribed The Term Deposit')
    fig = px.pie(df, values=df['target'].value_counts(), 
             names=['Do not subscribed term deposit','Subscribed term deposit'], title='Clients')
    st.plotly_chart(fig)

    # Sort the job categories
    st.write('#### Sort The Job Categories by Balance and Loan')
    col1, col2 = st.columns(2)
    fig1=px.bar(df.groupby('job_cat')['balance'].sum().sort_values(), orientation='h')
    fig2=px.bar(df.groupby('job_cat')['loan'].count().sort_values(), orientation='h')
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)

    #Check how many samples we have with default and no-default cases categorized by sex
    st.write('#### Age Distribution')
    fig=plt.figure(figsize=[20, 5])
    ax=sns.countplot(x = 'age', hue = 'target', data = df)
    ax.legend(labels = ['Do not subscribed term deposit', 'Subscribed term deposit'])
    st.pyplot(fig)
    
    # Plot of limit_balance group by sex, marital_status, and education_level
    st.write('#### Balance Distribution By Marital Status and Default Payment')
    marital = st.checkbox('marital')
    default = st.checkbox('default')

    if marital:
        fig = plt.figure(figsize=(4,4))
        sns.boxplot(x = 'marital', y = 'balance', data = df, showfliers = False)
        st.pyplot(fig)
    elif default:
        fig = plt.figure(figsize=(4,4))
        sns.boxplot(x = 'default', y = 'balance', data = df, showfliers = False)
        st.pyplot(fig)  

    # Scatter plot of high_blood_pressure status and death event
    st.write('#### Scatter Plot of Duration Call Versus Client Who Subsribed The Term Deposit')
    fig = px.scatter(df, x='duration', y="y")
    st.plotly_chart(fig)







if __name__ == '__main__':
    run()