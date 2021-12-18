import datetime
import io
import time
from concurrent import futures

import cv2
import grpc
import numpy as np
import tensorflow as tf
from mss import mss
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from win32api import GetSystemMetrics

import ImageClassification_pb2 as pb2
import ImageClassification_pb2_grpc as pb2_grpc
from model_rox import FacialExpressionModel

model = FacialExpressionModel("model.json", "model_weights.h5")

class ImageClassificationService(pb2_grpc.ImageClassificationServicer):
    def __init__(self, *args, **kwargs):
        pass

    def PredictImage(self, request, context):
        imageStream = io.BytesIO(request.file)
        img = Image.open(imageStream)
        img = img.resize((48, 48), Image.ANTIALIAS)
        img = img.convert("L")
        _image = np.asarray(img)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')
        label = model.predict_rox(_image[np.newaxis, :, :, np.newaxis])
        print(st + " | Prediction: ", label)
        result = {'result': label}
        return pb2.ImageResponse(**result)


def serve():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ImageClassificationServicer_to_server(
        ImageClassificationService(), server)
    server.add_insecure_port('[::]:31700')
    server.start()
    print(st + " | Started")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
