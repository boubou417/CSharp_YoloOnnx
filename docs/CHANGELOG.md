# 版本紀錄

## V1.5.11 Generic Shape Scoring Test

- Replace the wide 10-pixel binary hit test with a continuous, bidirectional centerline-distance score.
- Remove tiny foreground components before normalization so template speckles do not stretch the comparison bounds.
- Thin both the drawing and the selected template to one-pixel centerlines so stroke thickness does not dominate the result.
- Combine fine and broad distance falloff, strict two-way coverage, 90th-percentile outlier distance, and normalized path-length balance.
- Apply nonlinear score calibration to push unrelated scribbles lower while preserving high scores for close shapes.
- Derive every comparison feature from the currently selected image; no star-specific corners, intersections, or stroke order are hard-coded.
- Continue ignoring original position and size, drawing start point, drawing direction, and small hand-angle differences.
- Keep the existing green/orange/red score bands, CUDA/CPU fallback, locked yellow tracking, drawing storage, and animation behavior.

## V1.5.10 Locked Yellow Tracking Test

- Add explicit searching and locked marker states.
- Require connected red-tube and yellow-tip validation only for initial acquisition and true-loss reacquisition.
- After lock, track compact yellow candidates inside the velocity-predicted ROI without periodically requiring red geometry.
- Keep red support as a candidate scoring bonus while locked.
- Preserve prediction-distance ranking and two-frame fast-jump confirmation to prevent background yellow hijacking.
- Retain the lock through short misses and release it only after 450 ms of continuous yellow loss.
- Clear position, velocity, pending-jump, and color-pair state when the lock is released.
- Extend the WinForms tracking and stroke continuity grace to 500 ms so a recovered marker reconnects the trajectory.
- Preserve high-resolution geometry scaling, adaptive overlay sizing, CUDA/CPU fallback, scoring, and animation.

## V1.5.9 High-Resolution Tolerance Test

- Stop multiplying yellow and red connected-component minimum sample counts by total camera pixel area.
- Restore the proven 1.6-megapixel sample requirements for 5-megapixel and binned camera outputs.
- Continue scaling geometric distances, marker-size limits, red search radius, contact distance, and predictive ROI by resolution.
- Extend confirmed red-yellow tracking's temporary yellow-only allowance from 80 ms to 200 ms inside the predicted ROI.
- Extend raw tracking-loss lock retention from 120 ms to 240 ms.
- Extend the trajectory stroke-break delay from 300 ms to 450 ms so brief high-resolution misses can reconnect through interpolation.
- Keep strict red-yellow pairing for initial acquisition and full-frame reacquisition to protect against background false positives.
- Preserve fast-motion anti-jump, adaptive overlay sizing, CUDA/CPU fallback, scoring, and animation.

## V1.5.8 Resolution-Adaptive Tracking Test

- Use the same 1.6-megapixel reference scale for both rendering and marker tracking.
- Scale yellow hold tolerance, reacquisition radius, jitter dead zone, start movement distance, interpolation spacing, and fast-motion threshold.
- Scale elapsed-time jump allowance, maximum jump distance, and maximum marker speed in pixels per second.
- Scale predictive ROI minimum and maximum radius for higher-resolution frames.
- Scale red-tube search radius, yellow-to-red contact distance, minimum tube span, and maximum yellow component size.
- Scale connected-component sample requirements by image area to keep noise rejection consistent.
- Normalize candidate area and distance scores so 5-megapixel candidates are ranked like their 1.6-megapixel equivalents.
- Preserve V1.5.6 fast-motion anti-jump, V1.5.7 adaptive overlay sizing, CUDA/CPU fallback, scoring, and animation.

## V1.5.7 Resolution-Adaptive Overlay Test

- Use the 1.6-megapixel camera output as the visual-size reference.
- Compute an overlay scale from the actual frame pixel count and clamp it between 0.75 and 2.5.
- Keep trail thickness visually consistent when 5-megapixel frames are reduced to the same WinForms display area.
- Scale the yellow-tip marker, center dot, bounding box stroke, label font, and start/finish progress ring.
- Scale fingertip and thumb markers, pinch line, gesture progress ring, confidence text, and offsets.
- Scale person boxes, skeleton glow/core lines, joint markers, eye and torso lines, and the HELLO overlay.
- Leave WinForms controls and the external similarity-score panel unchanged.
- Preserve V1.5.6 fast-motion anti-jump, connected red-yellow validation, CUDA/CPU fallback, scoring, and animation.

## V1.5.6 Fast Motion Anti-Jump Test

- Hold a suspicious large marker jump as provisional instead of adding it to the trajectory immediately.
- Accept a provisional fast move on the next frame only when it remains near the candidate or continues in the same direction.
- Reject one-frame jumps that return toward the previous real marker position, preventing long zigzag interpolation lines.
- Scale jump, prediction-error, pending-match, and confirmed-step thresholds from the image diagonal for resolution-independent behavior.
- Increase predicted-position weighting when choosing between multiple yellow candidates.
- Prefer candidates with confirmed connected red-tube support over temporary yellow-only candidates.
- Do not refresh red occlusion grace or run a same-frame full-image fallback when a provisional jump is rejected.
- Preserve connected red-yellow validation, predictive full-resolution ROI tracking, CUDA/CPU fallback, scoring, and animation.

## V1.5.5 Connected Red-Yellow Pointer Test

- Replace the loose count of all nearby red pixels with connected-component analysis for the red tube.
- Reduce the red search radius from 110 to 60 pixels and require contact within 20 pixels of the yellow tip.
- Require the red component to extend at least 24 pixels from the yellow marker.
- Reject dense red/orange background patches using component fill ratio and principal-axis elongation checks.
- Narrow the accepted red/orange-red hue and saturation range to reduce warm wall, cardboard, and wood false positives.
- Refresh the temporary yellow-only allowance only when the final selected candidate actually has valid red-tube support.
- Shorten red-tube occlusion grace from 150 ms to 80 ms.
- Preserve predictive full-resolution ROI tracking, jitter tolerance, out-of-frame pause, CUDA/CPU fallback, scoring, and animation.

## V1.5.4 Red-Yellow Pointer Test

- Require the compact yellow tip to have an adjacent saturated red or orange-red tube segment.
- Reject oversized yellow/orange regions such as walls, clothing, and exhibition background objects.
- Search for the red tube within 110 pixels of each yellow candidate and require direct proximity plus a visible red run.
- Preserve full-resolution velocity-predicted ROI tracking for medium and fast pointer movement.
- Allow up to 150 ms of yellow-only tracking after a confirmed red-yellow pair so brief hand occlusion or motion blur does not interrupt the trajectory.
- Require a fresh red-yellow pair for initial acquisition and full-frame reacquisition, preventing jumps to unrelated yellow objects.
- Keep the existing jitter tolerance, start/finish holds, out-of-frame pause, scoring, animation, and CUDA/CPU fallback.

## V1.5.3 Predictive Yellow ROI Test

- Predict the next marker position from smoothed inter-frame velocity.
- Search a velocity-expanded ROI at full pixel resolution while the marker is tracked.
- Expand the ROI from a 140-pixel radius up to 420 pixels for fast movement.
- Fall back to the sampled full-frame search only when the predicted ROI does not contain a valid marker.
- Join yellow pixels across small two-pixel gaps to retain thin, motion-blurred marker streaks.
- Slightly relax brightness and saturation only inside the existing yellow hue range.
- Keep adaptive trajectory motion limits, interpolation, skin-tone rejection, and out-of-frame pause behavior.

## V1.5.2 Fast Yellow Tip Test

- Replace the fixed 160-pixel jump rejection with an elapsed-time adaptive motion allowance.
- Permit approximately 310 pixels per frame at 30 FPS and grow the allowance after a short missed frame, capped at 480 pixels.
- Preserve tracking through brief losses under 120 ms without requiring the marker to stop for reacquisition.
- Use stronger position response during fast motion and retain the 4-pixel jitter dead zone for stationary holds.
- Connect yellow components in eight directions and accept smaller motion-blurred components.
- Insert bounded intermediate trajectory points across sparse valid samples, with at most 16 inserted segments.
- Preserve HSV skin-tone rejection, out-of-frame pause behavior, and 30-pixel hold tolerance.

## V1.5.1 Stable Yellow Tip Test

- Replace broad RGB yellow detection with HSV hue and saturation filtering to reject skin tones.
- Lock a recently tracked marker and reject single-frame jumps larger than 160 pixels.
- Pause trajectory recording when the yellow marker leaves the image instead of switching to another yellow candidate.
- Require a returning marker to remain stable for 200 ms before tracking resumes.
- Increase start and finish hold tolerance from 14 to 30 pixels for a flexible, slightly shaking pointer.
- Add a 4-pixel dead zone and exponential position smoothing to reduce visible jitter.
- Start a new trajectory segment after a genuine marker loss so missing frames are never bridged by a long line.

## V1.5 Yellow Tip Tracking Test

- Replace fingertip landmark drawing with direct yellow marker tracking.
- Scan every third camera pixel and select the largest connected yellow component.
- Prefer candidates near the previous marker position to reduce jumps to background objects.
- Start drawing after the yellow tip remains still for 0.6 seconds.
- Require movement after starting, then finish after the tip remains still for 0.9 seconds.
- Trim the stationary finishing tail before saving and scoring the trajectory.
- Keep person boxes, skeleton rendering, trajectory saving, similarity scoring, and magic animation.
- Run yellow tracking independently of YOLO pose and TensorFlow Lite hand inference.

## Async Hand Landmark Performance Test

- Move TensorFlow Lite hand-landmark invocation from the camera/display thread to a single background task.
- Keep only one hand inference in flight and clone a new frame only when the worker is available.
- Prioritize the active drawing hand; alternate left and right requests while idle.
- Consume each completed hand result once, while permitting a confirmed result to remain visible for up to 400 ms.
- Wait briefly for the hand worker during Stop and Form close before disposing the TFLite interpreter.
- Keep stable landmark-only tracking thresholds and retain CAM/DISP/POSE/HAND diagnostics.

## Live Pipeline Diagnostics

- Add a WinForms-only diagnostics panel at the bottom-right; nothing is drawn over the camera image.
- Measure actual camera acquisition FPS and displayed-frame FPS over one-second windows.
- Show YOLO pose inference milliseconds and its equivalent maximum inference FPS.
- Measure the latest TensorFlow Lite hand-landmark invocation time.
- Show hand results as `FOUND`, `LOST`, `NO ROI`, `ERROR`, or `WAIT`.
- Reset all diagnostic counters whenever acquisition starts.
- Preserve stable landmark-only tracking, CUDA/CPU fallback, and the external similarity score panel.

## WinForms Similarity Score Panel

- Remove the shape-similarity panel from the camera bitmap.
- Add a fixed 300-pixel-wide score panel at the bottom-left of the WinForms window.
- Show `形狀相似度：--` before scoring and the numeric score after drawing.
- Use green, orange, or red score text according to the result.
- Keep drawing status text in the remaining bottom status area.

## V1.4 High Speed Tracking Rollback

- Reverted the expanded hand crop, lower hand-presence threshold, 40 ms request interval, dynamic jump allowance, and interpolated points after real-camera testing showed joint misidentification without improving fast-motion capture.
- Restored the stable 67 ms interval, 0.45 drawing confidence, original crop, and original jump rejection.
- Preserved CUDA-first/CPU-fallback selection and the independent display/pose threads.
- Fast motion will be addressed with a true inter-frame tracker or palm detector instead of relaxed landmark heuristics.

## V1.4 High Speed Tracking Test

- Increase fingertip inference from about 15 FPS (67 ms) to a maximum of about 25 FPS (40 ms) while drawing.
- Expand the active-hand search crop by 25% to tolerate delayed pose wrist coordinates.
- Reduce drawing hand-presence threshold from 0.45 to 0.40.
- Scale the accepted fingertip movement distance by elapsed sample time, up to three times the normal limit.
- Insert bounded intermediate trajectory points every 12 pixels, with at most 16 points per detected segment.
- Keep the existing 500 ms missing-fingertip stroke break so long tracking losses are not connected.
- Preserve CUDA-first and automatic CPU fallback behavior.

## V1.3 CUDA Auto Selection Test

- Replace the CPU-only ONNX Runtime package with `Microsoft.ML.OnnxRuntime.Gpu` 1.27.0.
- Try NVIDIA CUDA device 0 when the application starts.
- Automatically create a CPU session when CUDA, cuDNN, the driver, or the CUDA provider is unavailable.
- Show `CUDA GPU` or `CPU fallback` in the window title.
- Keep the fixed YOLO pose input at 640 x 640.

## V1.3 Performance Test

- Split camera acquisition/display and YOLO pose inference into independent background threads.
- Keep only the newest pending AI frame so slow inference cannot build a delayed frame backlog.
- Display live camera frames immediately while reusing the latest completed person boxes and skeleton.
- Keep the fixed 640 x 640 ONNX model input.
- Add live pose inference time (ms) to the on-screen diagnostics.
- Preserve the V1.2 drawing, fingertip, scoring, and magic-animation behavior.


此檔案記錄 CSharp_YoloOnnx 的重要版本變更。

## [V1.2] - 2026-08-03

### 新增

- 加入透明背景的藍紫色魔法陣 GIF 測試動畫。
- 有效軌跡成功儲存後，立即在即時相機畫面中央播放約 2 秒動畫。
- 動畫資源自動複製到執行檔旁的 `Assets` 資料夾。
- 加入黑底白線的單筆五角星 `Templates/template.png` 作為預設比對範本。
- 完成繪圖後自動正規化使用者軌跡與五角星範本，計算 `0～100` 相似度。
- 在即時畫面下方顯示大型「五角星相似度」分數面板。

### 測試條件

- 本版暫不判斷圖形相似度；只要軌跡達到最低點數且成功儲存就播放。
- 軌跡太短或儲存失敗時不播放動畫。
- 後續版本再將觸發條件改為相似度達到指定門檻。
- 五角星範本採用一筆連續路徑，尺寸、背景與線寬和軌跡輸出格式一致。
- 分數以雙向容許距離的像素精確率與召回率計算，並寫入該次軌跡 JSON。
- 分數顏色暫以 75／55 分區分，僅供測試觀察，尚不控制動畫播放。
- 繪圖階段的手部信心門檻提高到 45%，待機尋手仍維持 35%。
- 捏合加入 180 ms 預確認，單幀錯誤不會立刻進入完成倒數。
- 食指突然跳向拇指、但拇指本身幾乎未移動時，暫時保留上一個可靠食指位置。
- 比對明確改為只比較最後形狀，不使用起點、終點、方向或筆順。
- 依各自線條邊界等比例縮放並置中，原始繪圖大小與位置不納入評分。
- 線條容許距離由 6 px 放寬到 10 px，並在 ±8° 內嘗試旋轉後取最高分。

## [V1.1] - 2026-07-31

### 新增

- 加入 MediaPipe 21 點手部模型，以左右手食指指尖（landmark 8）繪圖。
- 加入張掌 0.6 秒開始、食指與拇指捏合 0.6 秒完成的差異化手勢。
- 在畫面上顯示青色食指定位標記與辨識信心值。
- 支援多筆畫；指尖短暫離開鏡頭後重新辨識時不會跨段連線。
- 新增 V1.1 測試指南。

### 調整

- 取消手腕高於肩膀的開始條件，左右任一手都能以張掌開始。
- 開始後鎖定選用的手，張掌與捏合分別使用綠色與橘色進度提示。
- 手部與手腕／手肘接受門檻降至 35%，指尖標記保留時間延長至 0.4 秒。
- 指尖中斷分段門檻由 0.3 秒延長至 0.5 秒，降低短暫漏偵測造成的軌跡斷裂。
- 移除胸前備用完成手勢，避免胸前與肩膀以下形成無法記錄軌跡的區域。
- 繪圖範圍改為整張相機影像；只以捏合 0.6 秒完成。
- 加入每隻手獨立的 0.3 秒位置快取；手肘漏偵測時沿用前臂方向，手腕短暫漏偵測時沿用上一個手部位置。
- 手部裁切範圍由前臂長度的 1.8 倍放大至 2.2 倍，模型輸入尺寸與推論次數維持不變。
- 捏合確認期間暫停記錄軌跡；放開會取消完成並以新筆畫繼續。
- 軌跡改由食指指尖產生，並提高移動反應速度。
- 即時預覽只保留最新畫面，避免取像期間 UI 佇列累積而延遲顯示骨架與人物方框。
- PictureBox 在收到最新畫面後立即重繪，並釋放每幀的 ONNX 與 GDI 資源。
- 將 Spinnaker 影像逐列複製到獨立 Bitmap，避免下一幀覆寫已畫好的即時疊圖。
- 即時畫面加入 `LIVE | YOLO persons` 診斷標記，方便區分顯示與人物偵測問題。
- 圖片與 JSON 儲存格式加入筆畫起點資料，供後續圖形比對使用。

## [V1.0.1] - 2026-07-31

### 新增

- 新增 Visual Studio 專案用的 `.gitignore`。
- 建立 `docs/CHANGELOG.md`。

### 維護

- 確認 `.vs`、`bin`、`obj` 未納入版本控制。
- 檢查 Solution 結構為單一 `CSharp_YoloOnnx` WinForms 專案。
- 確認專案目標為 .NET Framework 4.7.2。
- 確認 Debug／Release 均提供 Any CPU 與 x64 組態。
