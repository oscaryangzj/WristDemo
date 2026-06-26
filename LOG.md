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


## Hand landmark

```
input: (hand_image, rects)
[
    hand_image: after rotate and crop
    rects: [cx, cy, w, h, angle]
]
```

```aiignore
output: ((N, 21, 2), (N, 3))

21 coordination for each hand.
rotated_image_height, rotated_image_width, leftorright
```

resize and padding hands to (3, 224, 224) 

-> ONNX model to get key coordinations, handscore and leftorright

-> get the corresponding coordinations in the original image






