import streamlit as st   # pip install streamlit==1.17.0 streamlit-webrtc==0.44.2 altair==4.0
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import time
import redis
from collections import deque
import altair as alt
import random,os
import plotly.express as px
import json
from collections import Counter
import twilio
from twilio.rest import Client
import logging
import random
import configparser

logger = logging.getLogger(__name__)

# @st.experimental_memo
@st.cache_data
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """
    # Ref: https://www.twilio.com/docs/stun-turn/api

    stun_servers = ["stun.l.google.com:19302","stun1.l.google.com:19302","stun2.l.google.com:19302","stun4.l.google.com:19302",
        "stun01.sipphone.com","stun.ekiga.net","stun.fwdnet.net","stun.ideasip.com","stun.iptel.org","stun.rixtelecom.se","stun.schlund.de",
        "stunserver.org","stun.softjoys.com","stun.voiparound.com","stun.voipbuster.com", "stun.voipstunt.com","stun.voxgratia.org","stun.xten.com" ]

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    try:
        account_sid = config.get('Twilio', 'ACCOUNT_SID')
        auth_token = config.get('Twilio', 'AUTH_TOKEN')
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except (configparser.NoOptionError, twilio.base.exceptions.TwilioRestException):
        logger.warning(
            "Twilio credentials are not set or there was an error connecting to Twilio. Fallback to a free STUN server from Google."  # noqa: E501
        )
        chosen_stun_server = random.choice(stun_servers)
        return [{"urls": [f"stun:{chosen_stun_server}"]}]

# Create a Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)
st.set_page_config(page_title = 'AI camera monitor',layout="wide")
#============= main plot of model plugin ===============

def get_data_from_redis(webcams_num,window_que):
    total_faces = sum([int(r.get(f'global_face_count_{i}') or 0) for i in range(webcams_num)])
    window_que.append(total_faces)
    total_genders = sum((Counter(json.loads(r.get(f'global_gender_count_{i}') or '{"Male": 0, "Female": 0}')) for i in range(webcams_num)), Counter())
    total_ages = sum((Counter(json.loads(r.get(f'global_age_count_{i}') or '{"(0-6)": 0, "(6-16)": 0, "(16-25)": 0, "(25-35)": 0, "(35-43)": 0, "(43-53)": 0, "(53-65)": 0, "(65-100)": 0}')) for i in range(webcams_num)), Counter())
    return total_faces, total_genders, total_ages


def draw_face_bar(placeholder, webcams_num , window_que):
    total_faces, _, _ = get_data_from_redis(webcams_num, window_que)
    face_count_df = pd.DataFrame(dict(time=[i for i in range(len(window_que))], count=list(window_que)))
    face_count_fig = px.bar(face_count_df, x='time', y='count')
    face_count_fig.update_layout(showlegend=True, autosize=False, width=400, height=500)
    placeholder.write(face_count_fig)

def draw_gender_pie(placeholder, webcams_num, window_que ,accumulate=True):
    _, total_genders, _ = get_data_from_redis(webcams_num,window_que)
    if accumulate==False:
        total_genders = Counter(json.loads(r.get(f'global_gender_count_{webcams_num - 1}') or '{"Male": 0, "Female": 0}'))

    gender_count_fig = px.pie(values=total_genders.values(), names=total_genders.keys(), title='Gender Distribution')
    gender_count_fig.update_layout(showlegend=True, autosize=False, width=400, height=500)
    placeholder.write(gender_count_fig)

def draw_age_pie(placeholder, webcams_num, window_que, accumulate=True):
    _, _, total_ages = get_data_from_redis(webcams_num,window_que)
    if accumulate==False:
        total_ages = Counter(json.loads(r.get(f'global_age_count_{webcams_num - 1}') or '{"(0-6)": 0, "(6-16)": 0, "(16-25)": 0, "(25-35)": 0, "(35-43)": 0, "(43-53)": 0, "(53-65)": 0, "(65-100)": 0}'))

    age_count_fig = px.pie(values=total_ages.values(), names=total_ages.keys(), title='Age Distribution')
    age_count_fig.update_layout(showlegend=True, autosize=False, width=400, height=500)
    placeholder.write(age_count_fig)

#================= main plot of model plugin <end> ================

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 0, 255), int(round(frameHeight / 150)), 0)
    return frameOpencvDnn, faceBoxes

flip = st.checkbox("Flip")

class GlobalState:
    def __init__(self):
        self.state = {"selected_models": []}

    def update(self, new_state):
        self.state = new_state

    def get(self):
        return self.state


class DetectionProcessor(VideoProcessorBase):
    def __init__(self, global_state: GlobalState , webcam_idx: int):
        self.global_state = global_state
        self.webcam_idx = webcam_idx
        self.padding = 80
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-6)', '(6-16)', '(16-25)', '(25-35)', '(35-43)', '(43-53)', '(53-65)', '(65-100)']
        self.genderList = ['Male', 'Female']
        self.gender_count = {"Male": 0, "Female": 0}
        self.age_count = {age: 0 for age in self.ageList}  # Initialize age_count

        self.models = {}

        if "Face Detection" in self.global_state["selected_models"]:
            faceProto = "./model/opencv_face_detector.pbtxt"
            faceModel = "./model/opencv_face_detector_uint8.pb"
            self.models["Face Detection"] = cv2.dnn.readNet(faceModel, faceProto)

        if "Age Detection" in self.global_state["selected_models"]:
            ageProto = "./model/age_deploy.prototxt"
            ageModel = "./model/age_net.caffemodel"
            self.models["Age Detection"] = cv2.dnn.readNet(ageModel, ageProto)

        if "Gender Detection" in self.global_state["selected_models"]:
            genderProto = "./model/gender_deploy.prototxt"
            genderModel = "./model/gender_net.caffemodel"
            self.models["Gender Detection"] = cv2.dnn.readNet(genderModel, genderProto)

        if not self.models:
            pass
            # raise ValueError("No valid models selected.")

    def img_model_inference(self, frame):
        resultImg, faceBoxes = highlightFace(self.models["Face Detection"],
                                             frame) if "Face Detection" in self.models else (frame, [])

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - self.padding):
                         min(faceBox[3] + self.padding, frame.shape[0] - 1),
                   max(0, faceBox[0] - self.padding)
                   :min(faceBox[2] + self.padding, frame.shape[1] - 1)]

            try:
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            except:
                continue

            predictions = {}
            if "Gender Detection" in self.models:
                self.models["Gender Detection"].setInput(blob)
                genderPreds = self.models["Gender Detection"].forward()
                gender = self.genderList[genderPreds[0].argmax()]
                predictions["gender"] = gender
                self.gender_count[gender] += 1

            if "Age Detection" in self.models:
                self.models["Age Detection"].setInput(blob)
                agePreds = self.models["Age Detection"].forward()
                age = self.ageList[agePreds[0].argmax()]
                predictions["age"] = age

                if age not in self.age_count:
                    self.age_count[age] = 1
                else:
                    self.age_count[age] += 1

            label = ", ".join([f"{k}: {v}" for k, v in predictions.items()])
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
                        1, cv2.LINE_AA)

        return resultImg , faceBoxes

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")  # (480, 640, 3)
        img = img[::-1, :, :] if flip else img
        if "Face Detection" in self.models:
            img, faceBoxes = self.img_model_inference(img)
            global_face_count = len(faceBoxes)
            r.set(f'global_face_count_{self.webcam_idx}', global_face_count)
            r.set(f'global_gender_count_{self.webcam_idx}', json.dumps(self.gender_count))
            r.set(f'global_age_count_{self.webcam_idx}', json.dumps(self.age_count))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():

    webcams_num = st.sidebar.number_input('assign webcam streaming number', min_value = 1 ,max_value=4 ,value =1,step=1,key='int')
    GLOBAL_STATE = {"selected_models": []}
    GLOBAL_STATE["selected_models"] = st.sidebar.multiselect('Which models would you like to use?', ['Face Detection', 'Age Detection', 'Gender Detection'])

    streaming_cols = st.columns(webcams_num)
    det_processors = {}
    for i in range(webcams_num):
        with streaming_cols[i]:
            st.write(f"WebRTC Stream {i+1}")
            det_processors[i] = DetectionProcessor(GLOBAL_STATE,i)
            ctx = webrtc_streamer(key=f"camera_{i + 1}",
                                  video_processor_factory=lambda: det_processors[i],
                                  video_frame_callback=det_processors[i].recv,
                                  media_stream_constraints={"video": True, "audio": False},
                                  rtc_configuration={"iceServers": get_ice_servers()})

    accumulate = st.sidebar.checkbox('Accumulate data', value=True)
    WINDOW_SIZE = int(st.sidebar.number_input('que_windows(each data~0.5sec)', min_value = 1 ,max_value=120 ,value =60,step=1))
    window_que = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)

    button_cols = st.sidebar.columns(3)

    start_button  = button_cols[0].button("Start")
    stop_button   = button_cols[1].button("Stop")
    reset_button  = button_cols[2].button("Reset")

    chart_cols = st.columns(3,gap = 'small') #"small", "medium", or "large"
    placeholder1 = chart_cols[0].empty()
    placeholder2 = chart_cols[1].empty()
    placeholder3 = chart_cols[2].empty()

    if start_button or reset_button:
        if reset_button:
            r.flushall()
        while True:
            draw_face_bar(placeholder1, webcams_num,window_que)
            draw_gender_pie(placeholder2, webcams_num, window_que, accumulate)
            draw_age_pie(placeholder3, webcams_num, window_que, accumulate)
            time.sleep(0.5)
            if stop_button:
                break


main()
# if __name__ == '__main__':
#     main()
    #  wsl
    #  sudo service redis-server start
    #  streamlit run main.py
    #  streamlit run main.py --server.enableCORS true --server.address 192.168.99.253 --server.port 8501
    #  ssl-proxy-windows-amd64.exe -from 0.0.0.0:443 -to localhost:8501
    #  ssl-proxy-windows-amd64.exe -from 192.168.99.253:8501 -to localhost:8501
