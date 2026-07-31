# Hand-landmark model

`hand_landmark_full.tflite` comes from the Google MediaPipe Hands npm
package `@mediapipe/hands@0.4.1675469240`.

- Source: https://www.npmjs.com/package/@mediapipe/hands
- Model documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Package license: Apache-2.0
- SHA-256: `8c026882c9ec059ce0f8e75266bee5a9a23c341a40e0000df755374d3d1b9b68`

The application uses the existing YOLO wrist and elbow keypoints to build a
rotated hand crop, then reads landmark 8 (right index fingertip) from this
model. The palm detector from the complete MediaPipe pipeline is therefore not
included.
