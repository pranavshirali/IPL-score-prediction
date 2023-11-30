# Importing essential libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

# Loading the dataset
df = pd.read_csv("D:\\IPLScorePredictor\\ipl.csv")

# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)

# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

# Removing the first 5 overs data in every match
df = df[df['overs'] >= 5.0]

# Converting the column 'date' from string into datetime object
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

df.dropna(subset=['date'], inplace=True)
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

# Keep only relevant encoded columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',  
                         'bat_team_Kings XI Punjab','bat_team_Kolkata Knight Riders',
                         'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
                         'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
                         'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 
                         'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
                         'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
                         'bowl_team_Royal Challengers Bangalore', 
                         'bowl_team_Sunrisers Hyderabad',
                         'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
print("Encoded df shape:", encoded_df.shape) 


# Splitting the data into train and test set
X_train = encoded_df.drop(labels=['total', 'date'], axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels=['total', 'date'], axis=1)[encoded_df['date'].dt.year >= 2017]
# Validate shape
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Get number of columns 
num_cols = len(X_train.columns)
print("Number of columns:", num_cols)

# Update temp array length
temp_array = [0] * num_cols 


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# # Removing the 'date' column
# X_train.drop(labels='date', axis=True, inplace=True)
# X_test.drop(labels='date', axis=True, inplace=True)

# Linear Regression Model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)


# Streamlit app
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon=":bar_chart:",
    layout="wide"
)

# st.markdown(    
#     """
#     <style>
#         [data-testid="stAppViewContainer"]{
#         background-color: #e5e5f7;
# opacity: 0.8;
# background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #e5e5f7 10px ), repeating-linear-gradient( #444cf755, #444cf7 );
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.title("IPL Score Predictor")
st.markdown("Predicting the score for an IPL match based on machine learning!")

# User input features
st.sidebar.header("Enter the values")
selected_bat_team = st.sidebar.selectbox("Select Batting Team", consistent_teams)
selected_bowl_team = st.sidebar.selectbox("Select Bowling Team", consistent_teams)
overs = st.sidebar.slider("Overs", 5.0, 20.0, 10.0,step = 0.1)
runs = st.sidebar.number_input("Runs", min_value=0, step=1)
wickets = st.sidebar.number_input("Wickets", min_value=0, step=1)
runs_in_prev_5 = st.sidebar.number_input("Runs in Previous 5 Overs", min_value=0, step=1)
wickets_in_prev_5 = st.sidebar.number_input("Wickets in Previous 5 Overs", min_value=0, step=1)

# Predicting results
temp_array = [0] * len(X_train.columns)
  

# Initialize an array of zeros
temp_array[consistent_teams.index(selected_bat_team)] = 1
temp_array[8 + consistent_teams.index(selected_bowl_team)] = 1
temp_array[-5:] = [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
columns = ['bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
           'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
           'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
           'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
           'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
           'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
           'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5']
temp_array = np.array([temp_array], dtype=object)  # Convert to object dtype to allow different lengths
print("Shape of temp_array:", temp_array.shape)
print("Length of columns:", len(columns))

temp_df = pd.DataFrame(data=temp_array, columns=columns)


# Predict the score
predicted_score = int(linear_regressor.predict(temp_df)[0])

# Display the predicted score
st.subheader("Predicted Score")
if predicted_score > 200:
    st.success(f"The predicted score for {selected_bat_team} batting against {selected_bowl_team} is: {predicted_score} - Wow! That's a high score!")
elif predicted_score > 150:
    st.success(f"The predicted score for {selected_bat_team} batting against {selected_bowl_team} is: {predicted_score} - Looks like a good score.")
else:
    st.success(f"The predicted score for {selected_bat_team} batting against {selected_bowl_team} is: {predicted_score} - It might be a challenging total.")