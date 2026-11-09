import requests

record = {"Location": "Launceston",
 "WindGustDir": "SE",
 "WindDir9am": "ENE",
 "WindDir3pm": "N",
 "RainToday": "No",
 "MinTemp": 0.0,
 "MaxTemp": 13.0,
 "Rainfall": 0.0,
 "WindGustSpeed": 22.0,
 "WindSpeed9am": 7.0,
 "WindSpeed3pm": 6.0,
 "Humidity9am": 81.0,
 "Humidity3pm": 49.0,
 "Pressure9am": 1020.2,
 "Pressure3pm": 1017.3,
 "Temp9am": 6.2,
 "Temp3pm": 12.4}

# url = "http://localhost:9696/predict"   # local (python predict.py or docker run)
url = "http://<env-name>.eba-????????.eu-north-1.elasticbeanstalk.com/predict"   # AWS Elastic Beanstalk

response = requests.post(url, json=record)

print(response)

print(response.json())