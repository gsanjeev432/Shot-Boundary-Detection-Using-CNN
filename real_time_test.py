
import ffmpeg
import numpy as np
import tensorflow as tf

from cnn_model import ModelParams, TransNet
from model_utils import scenes_from_predictions

input_video = 'anime.mp4'
# initialize the network
params = ModelParams()
params.CHECKPOINT_PATH = "./model/trained_model"

net = TransNet(params)

# export video into numpy array using ffmpeg
video_stream, err = (
    ffmpeg
    .input(input_video)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(params.INPUT_WIDTH, params.INPUT_HEIGHT))
    .run(capture_stdout=True)
)
video = np.frombuffer(video_stream, np.uint8).reshape([-1, params.INPUT_HEIGHT, params.INPUT_WIDTH, 3])

# predict transitions using the neural network
predictions = net.predict_video(video)

import cv2

cap = cv2.VideoCapture(input_video)
count = 0
frame_width = 1024
frame_height = 768
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while(cap.isOpened()):
    ret, frame = cap.read()
    if(not ret): break

    frame= cv2.resize(frame, (frame_width,frame_height))
    if predictions[count] > 0.1:
        frame = cv2.putText(frame, 'Transition Detected', (100,100), cv2.FONT_HERSHEY_SIMPLEX ,
                            2, (0, 255, 0), 4, cv2.LINE_AA)
        print("Transition detected at frame no - {}".format(count))
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        
    count += 1
    out.write(frame)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Generate list of scenes from predictions, returns tuples of (start frame, end frame)
scenes = scenes_from_predictions(predictions, threshold=0.1)

print(scenes)
