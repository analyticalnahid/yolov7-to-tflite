import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('tfmodel/')
tflite_model = converter.convert()

with open('tfmodel/yolov7_model.tflite', 'wb') as f:
	f.write(tflite_model)
