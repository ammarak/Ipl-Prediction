# Importing Essential Libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Linear Regression Model
filename = 'iplmodel.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods = ['POST'])
def predict():
    temp_array = list()

    if request.method == "POST":

        runs  = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        overs = int(request.form['overs'])
        run_last_5 = int(request.form['runs_last_5'])
        wickets_last_5 = int(request.form['wickets_last_5'])
        striker = int(request.form['striker'])
        non_striker = int(request.form['non-striker'])
        year = int(request.form['year'])

        temp_array = temp_array + [runs, wickets, overs, run_last_5, 
        wickets_last_5, striker, non_striker, year]
        
        
        batting_team = request.form['batting-team']
        if batting_team == "bat_team_Delhi Daredevils":
            temp_array = temp_array + [1,0,0,0,0,0,0]
        elif batting_team == 'bat_team_Kings XI Punjab':
            temp_array = temp_array + [0,1,0,0,0,0,0]
        elif batting_team == "bat_team_Kolkata Knight Riders":
            temp_array = temp_array + [0,0,1,0,0,0,0]
        elif batting_team == "bat_team_Mumbai Indians":
            temp_array = temp_array + [0,0,0,1,0,0,0]
        elif batting_team == "bat_team_Rajasthan Royals":
            temp_array = temp_array + [0,0,0,0,1,0,0]
        elif batting_team == "bat_team_Royal Challengers Bangalore":
            temp_array = temp_array + [0,0,0,0,0,1,0]
        elif batting_team == "bat_team_Sunrisers Hyderabad":
            temp_array = temp_array + [0,0,0,0,0,0,1]
        elif batting_team == "bat_team_Chennai Super Kings":
            temp_array = temp_array + [0,0,0,0,0,0,0]
    
        
        bowling_team = request.form['bowling-team']
        if bowling_team == "bowl_team_Delhi Daredevils":
            temp_array = temp_array + [1,0,0,0,0,0,0]
        elif bowling_team == 'bowl_team_Kings XI Punjab':
            temp_array = temp_array + [0,1,0,0,0,0,0]
        elif bowling_team == "bowl_team_Kolkata Knight Riders":
            temp_array = temp_array + [0,0,1,0,0,0,0]
        elif bowling_team == "bowl_team_Mumbai Indians":
            temp_array = temp_array + [0,0,0,1,0,0,0]
        elif bowling_team == "bowl_team_Rajasthan Royals":
            temp_array = temp_array + [0,0,0,0,1,0,0]
        elif bowling_team == "bowl_team_Royal Challengers Bangalore":
            temp_array = temp_array + [0,0,0,0,0,1,0]
        elif bowling_team == "bowl_team_Sunrisers Hyderabad":
            temp_array = temp_array + [0,0,0,0,0,0,1]
        elif bowling_team == "bowl_team_Chennai Super Kings":
            temp_array = temp_array + [0,0,0,0,0,0,0]
        
        venue_mean_encode = float(request.form['venue_mean_encode'])
        batsman_mean_encode = float(request.form['batsman_mean_encode'])

        temp_array = temp_array + [venue_mean_encode, batsman_mean_encode]


        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])

        return render_template('index.html', my_prediction = f"The Score Should be between {my_prediction-10} to {my_prediction+5} ", data = temp_array)


if __name__ == '__main__':
    app.run(debug = True)
    





            
	 