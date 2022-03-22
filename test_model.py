import json
import cv2
import math
import numpy as np
import ffmpeg
import tensorflow as tf
from prettytable import PrettyTable

from cnn_model import ModelParams, TransNet
from model_utils import scenes_from_predictions

params = ModelParams()
params.CHECKPOINT_PATH = "./model/trained_model"

net = TransNet(params)

input_video = "5cUViKlaXS8.mp4"
video_list = json.load(open('test.json'))
tp, tn, fp, fn = 0, 0, 0, 0

for videoname, labels in video_list.items():
    if videoname != input_video:
        continue
    print('\nProcessing {}'.format(videoname))
    transitions = video_list[videoname]['transitions']
    total_transitions = len(transitions)
    no_of_frames = video_list[videoname]['frame_num']

    video_stream, err = (
        ffmpeg
        .input('test/' + videoname)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(params.INPUT_WIDTH, params.INPUT_HEIGHT))
        .run(capture_stdout=True, quiet=True)
    )
    video = np.frombuffer(video_stream, np.uint8).reshape(
        [-1, params.INPUT_HEIGHT, params.INPUT_WIDTH, 3])

    predictions = net.predict_video(video)

    scenes = scenes_from_predictions(predictions, threshold=0.1)
    print(scenes)

    for begin, end in transitions:
        if begin in scenes and end in scenes:
            tp += 1
        elif begin not in scenes and end not in scenes:
            fn += 1

    for begin, end in scenes:
        if begin not in np.array(transitions) and end not in np.array(transitions):
            fp += 1

    break
    tn = total_transitions - (tp + fn + fp)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

f1_score = 2 * tp / (2 * tp + fp + fn)
accuracy = (tp + tn)/(tp + tn + fp + fn)

classification_report = PrettyTable()
classification_report.field_names = ["Video","#T","TP","FP","FN","Precision","Recall","F1"]
classification_report.add_row([input_video,total_transitions,tp,fp,fn,precision,recall,f1_score])
print(classification_report)
