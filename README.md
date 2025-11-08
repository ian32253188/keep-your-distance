
# Keep‑Your‑Distance  
> 一個用於監測社交距離的 Python 專案

## 目錄  
1. 專案說明  
2. 功能特色  
3. 系統需求  
4. 安裝與快速執行  
5. 使用範例  
6. 結構說明  
7. 貢獻方式  
8. 授權條款  

---

## 1. 專案說明  
此專案「Keep‑Your‑Distance」是一支使用 Python 撰寫的程式，用以偵測輸入視訊／影像中的人與人之間距離並提醒過近情況。  
適用於公共空間監控、活動場館安全、社交距離提示等應用場景。  

---

## 2. 功能特色  
- 實時讀取／處理影像或影片檔案。  
- 偵測人員位置並計算彼此間距離。  
- 標記距離過近的情況並輸出警示。  
- 支援 demo.py 快速測試。  

---

## 3. 系統需求  
- Python 3.x  
- 安裝必要套件：  
  ```bash
  pip install -r requirements.txt
  ```  
  （請在 requirements.txt 中列出：如 OpenCV、NumPy 等）  
- 可使用攝影機或影片作為輸入來源  

---

## 4. 安裝與快速執行  
1. 取得原始碼  
   ```bash
   git clone https://github.com/ian32253188/keep-your-distance.git
   cd keep-your-distance
   ```  
2. 安裝套件  
   ```bash
   pip install -r requirements.txt
   ```  
3. 執行 demo  
   ```bash
   python demo.py
   ```  
4. (如適用) 修改 config 設定，例如：影像來源路徑、距離門檻值、輸出路徑等。  

---

## 5. 使用範例  
假設您有一段影片 video.mp4 ，可用如下命令執行：  
```bash
python demo.py --input video.mp4 --threshold 1.5
```  
程式將讀取 video.mp4 、偵測畫面中人員、計算距離、並將標記結果輸出至 output.mp4 或顯示於螢幕。  

---

## 6. 專案結構說明  
```
keep-your-distance/
│
├── demo.py              # 主程式
├── distance_detector.py  # 偵測與距離計算模組
├── utils.py              # 工具函式與設定
├── requirements.txt      # Python 套件需求
└── README.md             # 本說明檔
```  

---

## 7. 貢獻方式  
歡迎 Pull Request 或 Issue 回報建議。  
- 請先於 Issue 討論新增功能或修正錯誤。  
- 建議使用 feature/xxx 或 bugfix/xxx 命名 branch。  
- 保持程式碼風格一致，並撰寫必要註解與文件。  

---

## 8. 授權條款  
本專案採 [MIT License](https://opensource.org/licenses/MIT) 授權。  
歡迎自由使用、修改與散佈，但需保留原始作者署名與授權檔案。  
