# Conversation Log

## 1. Initial request
**User:** 請幫我建立一個簡單的 AI vs Human 文章分類工具。要求：使用者輸入文本即顯示 AI% / Human%；可用 sklearn/transformers/自建特徵；使用 Streamlit；含可視化或統計量。  
**Assistant:** 建立了 Streamlit 分類器 `app.py`、加入簡易特徵與小型訓練樣本，提供即時機率、特徵表、最近推論；新增 `requirements.txt`，更新 README 使用說明。

## 2. Buttons not working; need more analysis
**User:** ai sample 跟 human sample 按鈕沒有用。統計後需要有分析，介面很好可維持。  
**Assistant:** 修正按鈕使用 `st.session_state` + `st.rerun`，增加 `generate_feature_insights` 顯示特徵解讀。

## 3. Buttons still problematic
**User:** 按鈕還是有問題。  
**Assistant:** 明確指定 `text_area` value 綁定 session state，按鈕使用 `on_click` callback `_set_sample_text` + `st.rerun`，解決覆寫問題。

## 4. Yellow warning; random samples; better examples
**User:** 有黃色框框不應該出現；範例沒隨機；範例不精確。  
**Assistant:** 去除 rerun 警告來源；範例改隨機池並增補更貼近的 AI/Human 例句。

## 5. Chinese handling & submit button
**User:** 中文判斷有問題，要中英都可判斷；AI 範例要有中文；需提交按鈕。  
**Assistant:** Token/句切分支援中英；加入中文 AI/Human 樣本與範例池；新增「開始判斷」按鈕控制推論；結果持久化。

## 6. Chinese bias toward Human; punctuation too high
**User:** 中文都判成 Human，punctuation_ratio 過高。  
**Assistant:** 修正標點計算（Unicode 類別、避免母數偏差），加入 CJK 比例與每句標點；擴充中文樣本；重寫檔案避免亂碼（後續訊息中因亂碼重新建立）。

## 7. Accuracy still low; Chinese weird; align samples
**User:** 中文較能判斷但準確率差；範例應與模型判斷一致。  
**Assistant:** 重建 `app.py`（移除亂碼），新增 hapax_ratio 特徵、balanced Logistic Regression、擴大中英文訓練樣本；範例按鈕直接抽取訓練集文本以對齊模型。

## 8. Feature analysis phrasing
**User:** 特徵分析不要明確說 AI 或人類，避免誤導。  
**Assistant:** 調整特徵觀察描述為中性風格，不再指明 AI/Human。

## 9. UX/layout improvements
**User:** 想提升使用者體驗排版。  
**Assistant:** 提出排版優化建議（卡片分區、CTA 配色、expander、歷史滾動、清空按鈕等）。

## 10. Apply layout tweaks
**User:** 請直接調整。  
**Assistant:** 實作排版改動：頂部流程提示、範例說明、結果區最高機率提示、特徵表放 expander、歷史加清空按鈕並限制高度，維持既有流程。
