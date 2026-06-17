# Overview

palm detection --> hand landmark --> gesture classifier

## Palm detection

```input: (h, w, 3) BGR image```
```
output: (N, 4):
[
    sqn_rr_size,
    rotation,
    sqn_rr_center_x,
    sqn_rr_center_y,
] 
```

resize and padding to (192, 192, 3) -> BGRtoRGB -> ONNX model to (N, 8) -> get box to (N,4)


