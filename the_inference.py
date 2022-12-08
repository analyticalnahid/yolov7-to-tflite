import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="tfmodel/yolov7_model.tflite")

# Name of the classes according to class indices.
names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Creating random colors for bounding box visualization.
colors = {
    name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)
}

img = cv2.imread("data/image2.jpg")
image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)
im = image.astype(np.float32)
im /= 255

 Allocate tensors.
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]["shape"]
interpreter.set_tensor(input_details[0]["index"], im)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]["index"])

ori_images = [img.copy()]

for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(output_data):
    image = ori_images[int(batch_id)]
    box = np.array([x0, y0, x1, y1])
    box -= np.array(dwdh * 2)
    box /= ratio
    box = box.round().astype(np.int32).tolist()
    cls_id = int(cls_id)
    score = round(float(score), 3)
    name = names[cls_id]
    color = colors[name]
    name += " " + str(score)
    cv2.rectangle(image, box[:2], box[2:], color, 2)
    cv2.putText(
        image,
        name,
        (box[0], box[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        [225, 255, 255],
        thickness=2,
    )
# plt.imshow(ori_images[0])
cv2.imshow("test", ori_images[0])
cv2.waitKey(0)

# Destroys all the windows created
cv2.destroyAllWindows()
