# 版本紀錄

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
