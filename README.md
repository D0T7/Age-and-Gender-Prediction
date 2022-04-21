# Age-and-Gender-Predection
Python program to predict age and gender of a person

In this I have tried to predict age and gender of a person from their facial feature with two different approaches.<br/>


The "age_and_gender_predection.py" requires ["age.prototxt","dex_chalearn_iccv2015.caffemodel", "gender.prototxt" and "gender.caffemodel"](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).<br/>

To run it just copy the above mentioned files to the code directory and run the py file from terminal with<br/> ```python age_and_gender_predection.py``` <br/>or run it from an IDE.<br/>
This approach uses caffeemodel to try to predict the age and gender.<br/><br/><br/>


In the second file "age-gender-prediction-real-time.py" we use pretrained weights from deep face library ["age_model_weights.h5"](https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5) and ["gender_model_weights.h5"](https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5) to try to predict the age and gender.<br/>

Download those two files and copy them to the code directory.
Run the py file from Terminal with <br/> ```python age-gender-prediction-real-time.py``` <br/> or with IDE or run the Jupyter Notebook "realtime.ipynb"(contains the same code).


