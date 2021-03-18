import cv2
import json
import tensorflow as tf
print(tf.__version__)
model_dir = "models/mnist_dnn"

print(model_dir)
# #얼굴판별 모델 로드
# model_pgu = tf.keras.models.model_from_json(open(model_dir + "/pgu_mnist_model_dnn_json.json").read())
# model_pgu.load_weights(model_dir + "/pgu_mnist_model_dnn_weights.h5")
# pgu_face_class_names_file = model_dir + "/pgu_mnist_model_dnn_names.json"
# with open(pgu_face_class_names_file, "r") as json_file:
#     pgu_face_class_names = json.load(json_file)

# print(pgu_face_class_names)