import streamlit as st
import pandas as pd
import random
import pickle
from sklearn.preprocessing import StandardScaler
#Title 

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://imageio.forbes.com/specials-images/imageserve/68236c31bde79ea6839e6df4/Exterior/960x0.jpg?format=jpg&width=960')

st.header('Model of housing prices to predict median house values in California',divider =True)
#st.subheader('''User Must Enter Given Value To Predict Price:
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Features 🏘️')
st.sidebar.image('https://img.jamesedition.com/listing_images/2025/03/14/17/46/52/5f7cb276-6603-4971-9e55-d7c2a9acaf3a/je/507x312xc.jpg')



temp_df = pd.read_csv('/Users/peeyush11/Downloads/california.csv')
random.seed(52)
all_values = []
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var=st.sidebar.slider(f'Select {i} Value',int(min_value),int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]

import time 

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))
value = 0
progress_bar = st.progress(value)
placeholder = st.empty()
placeholder.subheader('Predicting Price')

place = st.empty()
place.image('https://i.pinimg.com/originals/f5/27/0a/f5270acbc4b98112fcd520d2eea023de.gif',width = 120)

if price>0:
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
        
    body= (f'Predicted Median House Price ${round(price,2)} Thousand dollar')
    placeholder.empty()
    place.empty()
    #st.subheader(body)
    st.success(body)
else:
    body = 'Invalid House Features'
    st.warning(body)
    