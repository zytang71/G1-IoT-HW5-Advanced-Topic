# G1-IoT-HW5-Advanced-Topic

## AI vs Human 文章分類工具

一個使用 Streamlit 建立的簡易 AI/Human 文本分類器。輸入文字即可即時看到 AI% / Human% 判斷，並顯示文字統計特徵與近期推論紀錄。

### 快速開始
1. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```
2. 啟動介面：
   ```bash
   streamlit run app.py
   ```
3. 開啟瀏覽器並輸入欲檢測的文字。可點擊「填入 AI 範例」或「填入 Human 範例」快速試用。

### 功能
- 即時顯示 AI / Human 機率與長條圖。
- 顯示文字特徵（詞長、多樣性、標點比例、句長變異等）。
- 保留最近推論紀錄（最多 10 筆）。
- 內建小型樣本集，使用邏輯迴歸作為分類模型，結果僅供參考。
