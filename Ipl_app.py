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

        temp_array = temp_array + [runs, wickets, overs, run_last_5, 
        wickets_last_5, striker, non_striker]
        
        
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


        venue_mean_encode = request.form['venue_mean_encode']
        if venue_mean_encode == "M Chinnaswamy Stadium":
            venue = 166.969386038688
        elif venue_mean_encode == "MA Chidambaram Stadium":
            venue = 167.34102769971898
        elif venue_mean_encode == "Maharashtra Cricket Association Stadium":
            venue = 165.5199203187251
        elif venue_mean_encode == "Rajiv Gandhi International Stadium":
            venue = 157.69929601072747
        elif venue_mean_encode == "Sharjah Cricket Stadium":
            venue = 158.1048387096774
        elif venue_mean_encode == "Dubai International Cricket Stadium":
            venue = 148.99308755760367
        elif venue_mean_encode == "Sheikh Zayed Stadium":
            venue = 150.73923444976077
        elif venue_mean_encode == "Eden Gardens":
            venue = 157.10220903395978
        elif venue_mean_encode == "Feroz Shah Kotla":
            venue = 163.71937521937522
        elif venue_mean_encode == "Wankhede Stadium":
            venue = 168.81402278702893


        temp_array = temp_array + [venue]


        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])

        return render_template('index.html', my_prediction = f"The Score Should be between {my_prediction-5} to {my_prediction+7} ", data = temp_array)


if __name__ == '__main__':
    app.run(debug = True)
    





            
	 
