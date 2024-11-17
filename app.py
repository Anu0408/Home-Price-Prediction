from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
import os.path as path
import sys

parent_directory = os.path.abspath(path.join(__file__ ,"../../"))
sys.path.append(parent_directory)

from HomePrice.pipelines.prediction import PredictionPipeline

app = Flask(__name__, template_folder='web-app/templates') # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    #return render_template("web-app/templates/index.html")
    return render_template("index.html")

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            property_area =str(request.form['property_area'])
            psqft =float(request.form['psqft'])
            nbhk =int(request.form['nbhk'])
            pool =int(request.form['pool'])
            clubhouse =int(request.form['clubhouse'])
            mall =int(request.form['mall'])
            park =int(request.form['park'])
            gym =int(request.form['gym'])
            
       
         
            #data = [property_area,psqft,nbhk,pool,clubhouse,park,gym]
            #print("Data is : ")
            #print(data)
            data=[200,psqft,nbhk,pool,clubhouse,mall,park,gym]
           
            data = np.array(data).reshape(1, 8)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'Something is wrong'

    else:
        return render_template('index.html')
    
if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)    