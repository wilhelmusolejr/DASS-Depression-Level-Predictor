from flask import Flask, render_template, request
import pandas as pd
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
# depression = pd.read_csv("cleaned_data.csv")

@app.route('/')
def index():
  return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
    newdata=dict()
    newdata['Q3A'] = int(request.form['Q3A'])
    newdata['Q5A'] = int(request.form['Q5A'])
    newdata['Q10A'] = int(request.form['Q10A'])
    newdata['Q13A'] = int(request.form['Q13A'])
    newdata['Q16A'] = int(request.form['Q16A'])
    newdata['Q17A'] = int(request.form['Q17A'])
    newdata['Q21A'] = int(request.form['Q21A'])
    newdata['Q24A'] = int(request.form['Q24A'])
    newdata['Q26A'] = int(request.form['Q26A'])
    newdata['Q31A'] = int(request.form['Q31A'])
    newdata['Q34A'] = int(request.form['Q34A'])
    newdata['Q37A'] = int(request.form['Q37A'])
    newdata['Q38A'] = int(request.form['Q38A'])
    newdata['Q42A'] = int(request.form['Q42A'])
    
    newdata['TIPI1'] = int(request.form['TIPI1'])
    newdata['TIPI2'] = int(request.form['TIPI2'])
    newdata['TIPI3'] = int(request.form['TIPI3'])
    newdata['TIPI4'] = int(request.form['TIPI4'])
    newdata['TIPI5'] = int(request.form['TIPI5'])
    newdata['TIPI6'] = int(request.form['TIPI6'])
    newdata['TIPI7'] = int(request.form['TIPI7'])
    newdata['TIPI8'] = int(request.form['TIPI8'])
    newdata['TIPI9'] = int(request.form['TIPI9'])
    newdata['TIPI10'] = int(request.form['TIPI10'])
    
    newdata['education'] = int(request.form['education'])
    newdata['urban'] = int(request.form['urban'])
    newdata['gender'] = int(request.form['gender'])
    newdata['religion'] = int(request.form['religion'])
    newdata['orientation'] = int(request.form['orientation'])
    newdata['race'] = int(request.form['race'])
    newdata['married'] = int(request.form['married'])
    newdata['familysize'] = int(request.form['familysize'])
    newdata['age_group'] = int(request.form['age_group'])
    
    
    # raw 
    df = pd.DataFrame([newdata.values()],columns=list(newdata.keys()))
    
    exp_prediction = model.predict(df)
    
    if exp_prediction == 1:
      message = "Normal"
      return render_template('index.html', p=message)
    if exp_prediction == 2:
      message = "Mild"
      return render_template('index.html', p=message)
    if exp_prediction == 3:
      message = "Moderate"
      return render_template('index.html', p=message)
    if exp_prediction == 4:
      message = "Severe"
      return render_template('index.html', p=message)
    if exp_prediction == 5:
      message = "Extremely Severe"
      return render_template('index.html', p=message)

if __name__ == '__main__':
  app.run(debug=True)
  
  
