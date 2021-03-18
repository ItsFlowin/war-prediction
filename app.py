import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'model/WAR_pickle.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        age = flask.request.form['age']
        salary = flask.request.form['salary']
        contract = flask.request.form['contract']
        dl = flask.request.form['dl']
        wins = flask.request.form['wins']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[age, salary, contract, dl, wins]],
                                       columns=['age', 'salary', 'contract','dl', 'wins'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Age':age,
                                                     'Salary':salary,
                                                     'Contract Value':contract,
                                                     'DL Trips':dl,
                                                     'Team Win Percentage':wins},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()