# 版本紀錄

此檔案記錄 CSharp_YoloOnnx 的重要版本變更。

## [V1.1] - 2026-07-31

### 新增

- 加入 MediaPipe 21 點手部模型，以右手食指指尖（landmark 8）繪圖。
- 在畫面上顯示青色食指定位標記與辨識信心值。
- 支援多筆畫；指尖短暫離開鏡頭後重新辨識時不會跨段連線。
- 新增 V1.1 測試指南。

### 調整

- 保留 YOLO 姿勢模型的右手舉起 0.8 秒開始、手腕回到胸前 0.8 秒完成。
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
