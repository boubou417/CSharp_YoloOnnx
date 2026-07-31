# Hand-landmark model

`hand_landmark_full.tflite` comes from the Google MediaPipe Hands npm
package `@mediapipe/hands@0.4.1675469240`.

- Source: https://www.npmjs.com/package/@mediapipe/hands
- Model documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Package license: Apache-2.0
- SHA-256: `8c026882c9ec059ce0f8e75266bee5a9a23c341a40e0000df755374d3d1b9b68`

The application uses the existing YOLO wrist and elbow keypoints to build a
rotated hand crop. Landmark 8 (right index fingertip) draws the trajectory,
and the normalized distance between landmark 4 (thumb tip) and landmark 8
drives the pinch-to-finish gesture. The palm detector from the complete
MediaPipe pipeline is therefore not included.
