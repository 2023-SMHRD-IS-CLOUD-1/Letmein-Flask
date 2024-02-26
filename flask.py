#!/usr/bin/env python
# coding: utf-8

# In[4]:


import boto3
from botocore.exceptions import ClientError
import logging


# In[5]:


import cv2
import numpy as np
import urllib.request

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    return image


# In[6]:


from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv
import matplotlib.pyplot as plt

dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())

model = load_model(model_path)


# In[7]:


def gender_detect(image):
    if image is None:
        print("Could not read input image")
        return "false"

    face, confidence = cv.detect_face(image)

    classes = ['man','woman']

    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        face_crop = np.copy(image[startY:endY,startX:endX])

        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]

        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        if conf[0] > conf[1]:
            return classes[0]
        else:
            return classes[1]


# In[12]:


import cv2
import os
from roboflow import Roboflow

def segment(url):
    rf = Roboflow(api_key="D6ShAf5rsaLMZskjVW0H")
    project = rf.workspace().project("letmein-2ttv8")
    model = project.version(1).model
    
    result = model.predict(url).json()
    sorted_result = sorted(result['predictions'], key=lambda x:(x['x'], x['y']))
    
    return sorted_result


# In[13]:


# segment("https://d1nypumamskciu.cloudfront.net/2f2f33738eca6a19edf20eee735d0f54.jpg")


# In[16]:


import numpy as np
from matplotlib import pyplot as plt
import cv2
import mediapipe as mp

def fmp(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    BG_COLOR = (192, 192, 192)  # 회색
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:

        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return 0
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )
            
        return results.pose_landmarks.landmark


# In[17]:


def bodytype(url):
    image = url_to_image(url)
    seg = segment(url)
    
    seg_point = sorted(seg[0]['points'], key=lambda x:(x['y'], x['x']))
    mp_point = fmp(image)
    
    height, width, _ = image.shape
        
    left_shoulder = [mp_point[12].y*height, mp_point[12].x*width] 
    right_shoulder = [mp_point[11].y*height, mp_point[11].x*width] 
    left_waist = [mp_point[24].y*height, mp_point[24].x*width] 
    right_waist = [mp_point[23].y*height, mp_point[23].x*width] 
    
    left_length = left_waist[0] - left_shoulder[0]
    left_waist[0] = left_waist[0] - left_length * 0.2
    
    right_length = right_waist[0] - right_shoulder[0]
    right_waist[0] = right_waist[0] - right_length * 0.2
    
    len1 = 10000
    len2 = 10000
    y1 = 0
    y2 = 0
    
    for p in seg_point:
        y = int(p['y'])
            
        if abs(y-int(left_shoulder[0])) < len1:
            len1 = abs(y-int(left_shoulder[0]))
            y1 = y
                
        if abs(y-int(left_waist[0])) < len2:
            len2 = abs(y-int(left_waist[0]))
            y2 = y
        
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0
        
    lenx1 = 10000
    lenx2 = 10000
    lenx3 = 10000
    lenx4 = 10000
        
    for p in seg_point:
        x = int(p['x'])
               
        if abs(int(p['y']) - y1) < 100:
            if abs(x - left_shoulder[1]) < lenx1:
                lenx1 = abs(x - left_shoulder[1])
                x1 = x

            if abs(x - right_shoulder[1]) < lenx2:
                lenx2 = abs(x - right_shoulder[1])
                x2 = x
            
        if abs(int(p['y']) - y2) < 100:
            if abs(x - left_waist[1]) < lenx3:
                lenx3 = abs(x - left_waist[1])
                x3 = x

            if abs(x - right_waist[1]) < lenx4:
                lenx4 = abs(x - right_waist[1])
                x4 = x
                
    print(x1, x2, x3, x4)
                
    if x2-x1 == 0:
        return 0
    else:
        return (x4-x3)/(x2-x1)


# In[19]:


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def sendEmail(to_mail, mem_id):
    my_mail = "gkfzkseka@naver.com"
    pwd = "chlwlsrbs1!"
    to_mail = "gkfzkseka@naver.com"  

    msg = MIMEMultipart()
    msg['Subject'] = 'Letmein'  
    msg['From'] = my_mail
    msg['To'] = to_mail

    text = MIMEText("귀하의 아이디는 "+mem_id+"입니다.")
    msg.attach(text)

    smtp = smtplib.SMTP("smtp.naver.com", 587)
    smtp.starttls()
    smtp.login(user=my_mail, password=pwd)
    smtp.sendmail(my_mail, to_mail, msg.as_string())
    smtp.close()


# In[ ]:


from flask import Flask, request, make_response, redirect
from flask_cors import CORS
from flask import Flask, render_template, send_file
import cv2
from flask import Blueprint, request
from werkzeug.utils import secure_filename
from flask import Flask, jsonify
import os

bp = Blueprint('image', __name__, url_prefix='/image')

app = Flask(__name__)

CORS(app)

# post 방식 통신
# 아이디를 이메일로 전송
@app.route("/email", methods=["POST"]) 
def Email():
    msg = request.get_json()
    email = msg['email']
    user_id = msg['user_id']
    sendEmail(to_mail, mem_id)
    return 'success'

# post 방식 통신
@app.route("/upload", methods=["POST"]) 
def func2():
    url = 'https://d1nypumamskciu.cloudfront.net/img.jpg'
    image = cv2.imread('data/save.jpg')
        
    # 성별 분석. mas or woman을 반환
    gen = gender_detect(image)
        
    # 체형분석
    body = bodytype(url)
    result = '사다리꼴'
    if gen == 'man':
        if body < 0.85:
            result = '역삼각형'
        elif body < 0.9:
            result = '사다리꼴'
        elif body < 0.95:
            result = '직사각형'
        else:
            result = '삼각형'
    elif gen == 'woman':
        if body < 0.85:
            result = '역삼각형'
        elif body < 0.9:
            result = '모래시계형'
        elif body < 0.95:
            result = '직사각형'
        else:
            result = '삼각형'
        
    return jsonify({'gender' : gen, 'body' : result})

@app.route("/os")
def test():
    msg = request.get_json()
    image = msg['image']
    cloth = msg['cloth']
    
    os.system("python test.py")
    output = os.popen("python test.py").read()
    print(output)

if __name__ == "__main__" : # main method 역할! --> 서버를 구동시키는 부분
    app.run(host = "0.0.0.0", port = "5000")

