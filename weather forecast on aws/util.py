
import pickle
import json
import numpy as np

#Step 1: - Read your artifacts

_data_columns = None
_model = None
_model1 = None

def read_artifacts():
    global _data_columns
    global _model
    global _model1
    
    with open("./RainDateColumns.json","r") as f:
      _data_columns = json.load(f)['data-columns']

    with open('RainTodayPic.pickle','rb') as f:
      _model = pickle.load(f)

    with open('RainTomorrowPic.pickle','rb') as f:
      _model1 = pickle.load(f)




def show_feature_names():
    return _data_columns


def predict_RainTodayTomarrow(i_MinTemp,i_MaxTemp,i_Rainfall,i_Evaporation,i_Sunshine,i_WindGustSpeed,i_WindSpeed9am,i_WindSpeed3pm,i_Humidity9am,i_Humidity3pm,i_Pressure9am,i_Pressure3pm,i_Cloud9am,i_Cloud3pm,i_Temp9am,i_Temp3pm,i_Month,i_Day,i_Location,i_WindGustDir,i_WindDir9am,i_WindDir3pm):
  
  input = np.zeros(len(show_feature_names()))
  
  # index_i_cat = np.where(xtrain.columns == i_cat)[0][0]

  index_i_Location = _data_columns.index(i_Location.lower())
  index_i_WindGustDir = _data_columns.index(i_WindGustDir.lower())
  index_i_WindDir9am = _data_columns.index(i_WindDir9am.lower())
  index_i_WindDir3pm = _data_columns.index(i_WindDir3pm.lower())



  input[0] = i_MinTemp
  input[1] = i_MaxTemp
  input[2] = i_Rainfall
  input[3] = i_Evaporation
  input[4] = i_Sunshine
  input[5] = i_WindGustSpeed
  input[6] = i_WindSpeed9am
  input[7] = i_WindSpeed3pm
  input[8] = i_Humidity9am
  input[9] = i_Humidity3pm
  input[10] = i_Pressure9am
  input[11] = i_Pressure3pm
  input[12] = i_Cloud9am
  input[13] = i_Cloud3pm
  input[14] = i_Temp9am
  input[15] = i_Temp3pm
  input[16] = i_Month
  input[17] = i_Day


  input[index_i_Location] = 1
  input[index_i_WindGustDir] = 1
  input[index_i_WindDir9am] = 1
  input[index_i_WindDir3pm] = 1

  x=[_model.predict([input])[0],_model1.predict([input])[0]]
  
  if(x[0]==0 and x[1]==0):
      y="Willnotrain | &#215; | Willnotrain | &#215; | It will not rain today and  tommarrow"
  elif(x[0]==0 and x[1]==1):
      y="Willnotrain | &#215; | Willrain | &#10003; | It will not rain today, It will rain tommarrow"
  elif(x[0]==1 and x[1]==0):
      y="Willrain | &#10003; | Willnotrain | &#215; | It will rain today, It will not rain tommarrow"
  else:
      y="Willrain | &#10003;| Willrain | &#10003; | It will rain today and tommarrow"
      
      

  return y

read_artifacts()
print(show_feature_names())
print(len(show_feature_names()))



print(predict_RainTodayTomarrow(10.0,20.0,0.0,4.800000,12.50000,31.0,7.0,22.0,54.0,57.0,1019.800000,1018.700000,4.431161,4.49925,15.8,18.7,12,16,'Hobart','WindGustDir_SE','WindDir9am_SE','WindDir3pm_SE'))




