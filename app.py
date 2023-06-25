from flask import Flask,request,render_template #for initializing flask & render the html templates
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline #for data from prediction pipeline

#Defining application name
application=Flask(__name__)

#Name for the app
app=application

#Routing for home page 
@app.route('/')
def home_page():
    return render_template('index.html') #Rendering index template

#Routing for the predict page
@app.route("/predict", methods = ["GET", "POST"])
def predict_datapoint(): #method for get request
    if request.method == "GET":
        return render_template("form.html") #to render the form templated
    
    else: #method for post request
        
        #to initalize request from the form which is obtain
        data = CustomData(
            cap_shape = request.form.get("cap_shape"),
            cap_surface = request.form.get("cap_surface"),
            cap_color = request.form.get("cap_color"),
            bruises = request.form.get("bruises"),
            odor = request.form.get("odor"),
            gill_attachment = request.form.get("gill_attachment"),
            gill_spacing = request.form.get("gill_spacing"),
            gill_size = request.form.get("gill_size"),
            gill_color = request.form.get("gill_color"),
            stalk_shape = request.form.get("stalk_shape"),
            stalk_root = request.form.get("stalk_root"),
            stalk_color_above_ring = request.form.get("stalk_color_above_ring"),
            stalk_color_below_ring = request.form.get("stalk_color_below_ring"),
            stalk_surface_above_ring = request.form.get("stalk_surface_above_ring"),
            stalk_surface_below_ring = request.form.get("stalk_surface_below_ring"),
            veil_type = request.form.get("veil_type"),
            veil_color = request.form.get("veil_color"),
            ring_number = request.form.get("ring_number"),
            ring_type = request.form.get("ring_type"),
            spore_print_color = request.form.get("spore_print_color"),
            population = request.form.get("population"),
            habitat = request.form.get("habitat")            
        )
        
        #to predict output
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data) #to predict the final output
        
        if pred == 0: #For edible
            results = "Edible"
            return render_template("edible.html", final_result = results) #to render edible template, if result is edible
        elif pred == 1:
            results = "Poisonous"
            return render_template("poisonous.html", final_result = results) #to render poisonous template, if result is poisonous
        else:
            result = "Data not found"
            return render_template("results.html", final_result = results) #to render result template, if result found any error

#to initalize the app when run
if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000)