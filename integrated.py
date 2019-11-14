from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

#pose
import argparse
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

#pose
estimator = load_pretrain_model('VGG_origin')
action_classifier = load_action_premodel('Action/framewise_recognition.h5')

#pose
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0


# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)

while True:
    show, frame = camera.read()

    #pose
    if show:
        fps_count += 1
        frame_count += 1


    #reading the frame
    frame = imutils.resize(frame,width=400)


    


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    #pose
    humans = estimator.inference(frameClone)
    pose = TfPoseVisualizer.draw_pose_rgb(frameClone, humans)
    frameClone = framewise_recognize(pose, action_classifier)
    height, width = frameClone.shape[:2]

    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        print(preds)
 
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)

    if (time.time() - start_time) > fps_interval:
        # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
        realtime_fps = fps_count / (time.time() - start_time)
        fps_count = 0  # 帧数清零
        start_time = time.time()
    fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
    cv2.putText(frameClone, fps_label, (width-160, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    num_label = "Human: {0}".format(len(humans))
    cv2.putText(frameClone, num_label, (5, height-45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # 显示目前的运行时长及总帧数
    if frame_count == 1:
        run_timer = time.time()
    run_time = time.time() - run_timer
    time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
    cv2.putText(frameClone, time_frame_label, (5, height-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    #cv2.imshow('Action Recognition based on OpenPose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()