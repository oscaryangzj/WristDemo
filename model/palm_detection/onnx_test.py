from pathlib import Path
import onnxruntime as ort
import cv2
import numpy as np
from matplotlib.pyplot import imshow

current_dir = Path (__file__).parent
# print(current_dir)

model_path = current_dir / "palm_detection_full_inf_post_192x192.onnx"
session = ort.InferenceSession (str (model_path))

frame = cv2.imread("/Users/oscaryang/PycharmProjects/WristDemo/hand.png")
# cv2.imshow("original", frame)
# cv2.waitKey(0)
print (frame.shape)

# resize to 192 * 192
image = cv2.resize(frame, (192, 192))
# cv2.imshow("resized", image)
# cv2.waitKey(0)
print (image.shape)

# float
image = image.astype (np.float32) / 255.0

# normalization
image = image / 255.0

# HWC -> CHW
image = image.transpose (2, 0, 1)

# BGR -> RGB
image = image[: : -1]

# add batch dimension
image = np.expand_dims (image, axis = 0)

outputs = session.run(
    None,
    {"input": image}
)

print (image.shape)
print (outputs[0][1])

