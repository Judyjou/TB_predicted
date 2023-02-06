# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:33:00 2022

@author: Judy
"""

import os
import traceback
from flask import Flask, render_template, request, redirect
import numpy as np
from predictFunction import make_file, predict_mainPart
import json
from PIL import Image
from foundation import logger
from io import BytesIO
import base64

dl_path=os.getcwd()
static_path=os.path.join(dl_path,"static")
UPLOAD_FOLDER = os.path.join(static_path,"image")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','dcm'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index_view():
    return render_template('TB_predict.html')
    # return 'Hello World!!!!'

@app.route('/predict/',methods=['GET','POST'])
def predict():
    response='For CNN Prediction'
    return response

@app.route('/ImgUpload/', methods=['GET', 'POST'])
def save_img():
    try:
        json_data = request.json
        b64_img_string = json_data['img']
        logger.error(type(b64_img_string))
        real_data = b64_img_string.split(",")[1]
        imgdata = base64.b64decode(real_data)
        
        image_data = BytesIO(imgdata)
        logger.error(type(image_data))
        img_name = json_data['name']  
        img = Image.open(image_data)
        # img.save(UPLOAD_FOLDER + img_name)
        target_path=os.path.join(UPLOAD_FOLDER, img_name)
        img.save(target_path)
        result = predict_mainPart()
        logger.error(result)
        return result
        
    except:
        logger.error(traceback.format_exc())

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
if __name__=='__main__':
    app.run(debug=True,port=8000)
