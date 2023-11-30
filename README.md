>[!NOTE]
>In IPL, it is very difficult to predict the actual score because, in a moment of time, the game can completely turn upside down!
# IPL Score Predictor

A brief description of what this project does and who it's for

1. Data Loading and Cleaning:
    - The code starts by importing necessary libraries and loading the IPL dataset from a CSV file.
    - Unwanted columns such as 'mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker' are removed.
    - Inconsistent teams are filtered out, and data for the first 5 overs in each match is removed.

2. Data Splitting:
    - The dataset is split into training and testing sets based on the year.

3. Model Training and Evaluation:
    - Three regression models are trained: Linear Regression, Decision Tree Regression, and Random Forest Regression.
    - The models are evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).



## Dataset information


__1. Match Information:__
- Match ID (mid): A unique identifier for each match.

- Venue: The location where the match was played.

* Date: The date on which the match took place.

**2. Player Information:**  
- Batsman: The player who is currently batting.

- Bowler: The player who is currently bowling.

- Striker: The player who is currently facing the delivery.

- Non-striker: The player at the non-striker's end.

**3. Team Information:**
 - Batting Team (bat_team): The team that is currently batting.
 - Bowling Team (bowl_team): The team that is currently bowling.

**4. Match Statistics:**
- Overs: The number of overs bowled in the match.

- Runs: The total runs scored by the batting team.

- Wickets: The total number of wickets fallen.

**5. Inning Details:**
- Runs in the last 5 overs: The runs scored in the last 5 overs of the batting team.

- Wickets in the last 5 overs: The number of wickets fallen in the last 5 overs.

**6. Target Variable:**
- Total: The total runs scored in the entire match by the batting team.
> [!TIP]
> To view the dataset paste the following link in any browser.

```
https://1drv.ms/x/s!Apnc3nKjoTmfgrIVIPsnjB6v8oSHDw?e=6Pd4ga
```






## Get Started
1. Clone the repository:
```
https://github.com/pranavshirali/IPL-Score-Predictor.git
```
2. Run the IPL Score Predictor script:
```
python ipl_score_predictor.py
```
>[!NOTE]
>Make sure Python complier and below libraries are installed in your PC

Installation:
```
pip install pandas numpy scikit-learn
```
If you're using Jupyter Notebooks or a similar environment, you might also want to install the required libraries for data visualization if not already installed:
```
pip install matplotlib seaborn
```
    
