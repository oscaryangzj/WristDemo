import onnxruntime as ort

session = ort.InferenceSession("hand_landmark_sparse_Nx3x224x224.onnx")

print("Inputs:")
for i in session.get_inputs():
    print(i.name)
    print(i.shape)

print("Outputs:")
for o in session.get_outputs():
    print(o.name)
    print(o.shape)