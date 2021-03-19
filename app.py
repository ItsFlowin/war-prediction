import flask
import pickle
import pandas as pd
import numpy as np

# Use pickle to load in the pre-trained model
with open(f'model/WAR_Pickle.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    input_variables=pd.DataFrame()
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        salary = flask.request.form['salary']
        contract = flask.request.form['contract']
        wins = flask.request.form['wins']
        age = flask.request.form['age']
        dl = flask.request.form['dl']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[salary, contract, wins, age, dl]],
                                       columns=['salary', 'contract', 'wins', 'age', 'dl'],
                                       dtype=float,
                                       index=['input'])

        # Transform Data into proper format for the model
        input_variables['age'] = input_variables['age'].apply(lambda x: (x-18))
        input_variables['salary'] = input_variables['salary'].apply(lambda x:np.log(x))
        input_variables['contract'] *= 0.01 
        input_variables['wins'] *= 0.01

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Salary':salary,
                                                     'Percent Contract Complete':contract,
                                                     'Team Win Percentage':wins,
                                                     'Age':age,
                                                     'DL Trips':dl},
                                     result=prediction, form_reuse=input_variables,
                                     )

if __name__ == '__main__':
    app.run()