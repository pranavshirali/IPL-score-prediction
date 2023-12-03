# Importing essential libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

def load_and_preprocess_data(file_path):
    # Loading the dataset
    df = pd.read_csv(file_path)

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
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    #Dropping rows where the 'date' is missing.
    df.dropna(subset=['date'], inplace=True)

    # Converting categorical features using OneHotEncoding method
    encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

    # Keep only relevant encoded columns
    relevant_columns = ['date'] + [col for col in encoded_df.columns if col.startswith(('bat_team_', 'bowl_team_'))] + ['overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']
    encoded_df = encoded_df[relevant_columns]

    return encoded_df

def split_data(encoded_df):
    # Splitting the data into train and test set
    X_train = encoded_df.drop(labels=['total', 'date'], axis=1)[encoded_df['date'].dt.year <= 2016]
    X_test = encoded_df.drop(labels=['total', 'date'], axis=1)[encoded_df['date'].dt.year >= 2017]

    y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
    y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    # Linear Regression Model
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)
    return linear_regressor

def main():
    # Streamlit app
    st.set_page_config(
        page_title="IPL Score Predictor",
        page_icon=":bar_chart:",
        layout="wide"
    )

    st.title("IPL Score Predictor")
    st.markdown("Predicting the score for an IPL match based on machine learning!")

    # Load and preprocess data
    file_path = "D:\\Model Predictor\\ipl.csv"
    encoded_df = load_and_preprocess_data(file_path)

    # Split data
    X_train, X_test, y_train, y_test = split_data(encoded_df)

    # Train Linear Regression model
    linear_regressor = train_linear_regression(X_train, y_train)

    # User input features
    st.sidebar.header("Enter the values")
    selected_bat_team = st.sidebar.selectbox("Select Batting Team", encoded_df.columns[1:9])
    selected_bowl_team = st.sidebar.selectbox("Select Bowling Team", encoded_df.columns[9:17])
    overs = st.sidebar.slider("Overs", 5.0, 20.0, 10.0, step=0.1)
    runs = st.sidebar.number_input("Runs", min_value=0, step=1)
    wickets = st.sidebar.number_input("Wickets", min_value=0, step=1)
    runs_in_prev_5 = st.sidebar.number_input("Runs in Previous 5 Overs", min_value=0, step=1)
    wickets_in_prev_5 = st.sidebar.number_input("Wickets in Previous 5 Overs", min_value=0, step=1)

    # Predicting results
    temp_array = [0] * len(X_train.columns)

    # Initialize an array of zeros
    temp_array[encoded_df.columns.get_loc(selected_bat_team)] = 1
    temp_array[encoded_df.columns.get_loc(selected_bowl_team)] = 1
    temp_array[-5:] = [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]

    temp_df = pd.DataFrame(data=[temp_array], columns=X_train.columns)

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

if __name__ == "__main__":
    main()