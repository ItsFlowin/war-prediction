import flask
import pickle

# use pickle to load in the pre-trained model
with open(f'model/WAR_pickle', 'rb') as f:
    model =pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':

        age = flask.request.form['age']
        salary = flask.request.form['salary']
        contract = flask.request.form['contract']
        dl = flask.request.form['dl']
        wins = flask.request.form['wins']

        input_var = pd.DataFrame([[age, salary, contract, dl, wins]], columns=['age', 'salary', 'contract', 'dl', 'wins'], dtype=float)

        prediction = model.predict(input_var) [0]

        return flask.render_template('main.html',
        original_input={'Age':age, 'Salary':salary, 'contract':contract, 'dl':dl, 'wins':wins}, result=prediction,)

if __name__ == '__main__':
    app.run()