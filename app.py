import random
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="AI vs Human Text Classifier",
    page_icon="🧠",
    layout="wide",
)

# Initialize default text once so sample buttons can override it reliably.
if "input_text" not in st.session_state:
    st.session_state["input_text"] = (
        "I spent the afternoon finishing my report, then rewrote the introduction to make it clearer."
    )


# English + Chinese stopwords for rough style cues.
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "you",
    "的",
    "了",
    "在",
    "是",
    "我",
    "你",
    "他",
    "她",
    "我們",
    "你們",
    "他們",
    "也",
    "很",
    "都",
    "不",
    "就",
    "會",
    "嗎",
    "啊",
    "吧",
    "這",
    "那",
    "有",
    "沒有",
    "一個",
    "什麼",
}


def tokenize_words(text: str) -> List[str]:
    """Return tokens including English words and contiguous Chinese characters."""
    return re.findall(r"[A-Za-z']+|[\u4e00-\u9fff]+", text.lower())


def split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in re.split(r"[.!?。！？]+", text) if s.strip()]
    return parts if parts else [text.strip()]


def extract_features(text: str) -> Dict[str, float]:
    words = tokenize_words(text)
    word_count = len(words)
    unique_words = len(set(words))
    counts = Counter(words)
    hapax = sum(1 for _, c in counts.items() if c == 1)

    sentences = split_sentences(text)
    sentence_lengths = [len(tokenize_words(s)) for s in sentences]
    sentence_mean = np.mean(sentence_lengths) if sentence_lengths else 0.0
    sentence_std = np.std(sentence_lengths) if sentence_lengths else 0.0
    burstiness = sentence_std / sentence_mean if sentence_mean else 0.0

    chars = [c for c in text if not c.isspace()]
    punctuation = [c for c in chars if unicodedata.category(c).startswith("P")]
    cjk_chars = [c for c in chars if re.match(r"[\u4e00-\u9fff]", c)]
    uppercase = [c for c in chars if c.isupper()]
    digits = [c for c in chars if c.isdigit()]

    stopword_hits = sum(1 for w in words if w in STOP_WORDS)
    long_words = sum(1 for w in words if len(w) >= 7)

    avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0.0
    punctuation_ratio = len(punctuation) / max(len(chars) - len(punctuation), 1)
    punct_per_sentence = len(punctuation) / len(sentences) if sentences else 0.0
    uppercase_ratio = len(uppercase) / len(chars) if chars else 0.0
    digit_ratio = len(digits) / len(chars) if chars else 0.0
    stopword_ratio = stopword_hits / word_count if word_count else 0.0
    lexical_diversity = unique_words / word_count if word_count else 0.0
    long_word_ratio = long_words / word_count if word_count else 0.0
    cjk_ratio = len(cjk_chars) / len(chars) if chars else 0.0
    hapax_ratio = hapax / word_count if word_count else 0.0

    return {
        "avg_word_length": avg_word_length,
        "lexical_diversity": lexical_diversity,
        "stopword_ratio": stopword_ratio,
        "punctuation_ratio": punctuation_ratio,
        "punctuation_per_sentence": punct_per_sentence,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "sentence_mean_len": sentence_mean,
        "sentence_std_len": sentence_std,
        "burstiness": burstiness,
        "long_word_ratio": long_word_ratio,
        "cjk_ratio": cjk_ratio,
        "hapax_ratio": hapax_ratio,
    }


def build_training_samples() -> List[Tuple[str, str]]:
    """Small seed dataset to let the classifier learn a rough boundary."""
    ai_samples = [
        "As an AI language model, I can provide a structured explanation of the topic with clear bullet points and a concise summary.",
        "Below is a step-by-step guide. First, initialize the environment; second, load the data; finally, evaluate the metrics.",
        "I do not possess consciousness or personal opinions. However, I can simulate a reasoning process to help you decide.",
        "The experiment demonstrates that increasing context length improves coherence in downstream tasks such as summarization.",
        "Here are three actionable suggestions: 1) refactor the helper function, 2) add unit tests, and 3) document the interface.",
        "作為一個模型，我可以提供問題拆解與步驟化的解法，並列出需要注意的邊界條件與假設。",
        "下方是簡要結論：模型在 15 個 epoch 後收斂，驗證集指標穩定，推論時間可控。",
        "此方法透過注意力機制捕捉長距關係，在摘要任務上優於傳統序列模型。",     
        "這份報告整理了實驗結果，顯示在高溫條件下效能仍維持穩定，適合部署在邊緣裝置。",
        "總結而言，系統在低資源環境下仍維持 92% 的 F1 分數，推理延遲控制在 80ms 以內。",
        "Data preprocessing includes normalization, deduplication, and linguistic filtering to reduce downstream noise.",
        "上班搭捷運時看到一位小朋友把座位讓給老人家，旁邊的人都微笑點頭，心情突然變得很好。",
        "The model outputs show strong convergence after 20 epochs, which suggests the optimizer found a stable minimum.",
    ]
    human_samples = [
        "模型沒有主觀意識，但能根據輸入條件模擬決策流程，提供可採取的行動建議。",
        "以下流程可加速開發：先生成合成數據，再用真實數據微調，最後以混合集做驗證。",
        "The model automatically generates a concise brief, highlights the top risks, and recommends three mitigation steps.",
        "I wrote this late at night while drinking coffee, so excuse the rough edges and occasional rambling in the middle.",
        "When I tried the recipe the first time, the dough wouldn't rise, but after leaving it near the stove it finally worked.",
        "Back in college we used to argue for hours about music, then crash on the couch and play whatever old records we had.",
        "The kids ran through the park, laughing as the sprinklers came on and soaked everyone before the sun went down.",
        "I missed the bus, so I walked home in the rain and thought about the conversation we'd had the day before.",
        "It felt like a long day at work, but the moment I opened the window the breeze made everything a little easier.",
        "昨天晚飯後出門散步，街角麵包店還在烤麵包，整條巷子都是剛出爐的香味。",    
        "週末和朋友去山上走步道，結果下了小雨，樹葉被打濕後反而更綠，空氣裡都是泥土味。",
        "凌晨趕稿時聽到外面有烏鴉叫，才發現自己熬得太晚，決定先睡覺明天再寫。",
        "早上擠公車時被雨滴打到，司機開了一個大轉彎，大家差點站不穩，還好沒人受傷。",
        "上週末回老家，幫爸媽整理院子，翻土時竟然挖出以前埋的玻璃彈珠。",
        "朋友臨時約吃火鍋，結果排隊排了快一小時，大家餓到先買炸雞邊等邊吃。",
        "昨晚睡前看了一本舊漫畫，翻到以前摺起來的頁角，才想起那時候卡在那段劇情。",
        "He missed his train, wandered around the station, and ended up chatting with a stranger until the next one arrived.",
        "I biked along the river, stopped for a cheap sandwich, and watched the sunset turn the buildings orange.",
        "在夜市買了一杯柳橙汁，老板加了好多冰塊，喝到最後還是酸酸甜甜的很好喝。",
        "這個方法的限制在於需要大量上下文，如果截斷過短，模型可能忽略關鍵細節。",
        "模型對輸入格式敏感，建議保持一致的段落結構與標點，以獲得較穩定的輸出。",
    ]

    samples = [("ai", t) for t in ai_samples] + [("human", t) for t in human_samples]
    return samples


@st.cache_resource(show_spinner=False)
def train_model() -> Tuple[any, List[str]]:
    data = build_training_samples()
    feature_rows = []
    labels = []
    for label, text in data:
        feature_rows.append(extract_features(text))
        labels.append(label)

    feature_names = list(feature_rows[0].keys())
    X = pd.DataFrame(feature_rows)
    y = np.array(labels)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=800, C=2.5, solver="lbfgs", class_weight="balanced"),
    )
    model.fit(X, y)
    return model, feature_names


def predict_text(model, feature_names: List[str], text: str) -> Dict[str, float]:
    features = extract_features(text)
    X = np.array([[features[name] for name in feature_names]])
    proba = model.predict_proba(X)[0]
    label_order = list(model.classes_)
    return {label_order[i]: float(proba[i]) for i in range(len(label_order))}


def render_probability_bar(probabilities: Dict[str, float]) -> None:
    prob_df = (
        pd.DataFrame(
            {
                "Label": list(probabilities.keys()),
                "Probability": [probabilities[k] * 100 for k in probabilities],
            }
        )
        .set_index("Label")
        .sort_values(by="Probability", ascending=False)
    )
    st.bar_chart(prob_df, height=200)


def render_features_table(feature_map: Dict[str, float]) -> None:
    df = pd.DataFrame(
        {
            "Feature": list(feature_map.keys()),
            "Value": [round(v, 4) for v in feature_map.values()],
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def add_to_history(text: str, probabilities: Dict[str, float]) -> None:
    history = st.session_state.setdefault("history", [])
    predicted_label = max(probabilities, key=probabilities.get)
    history.append(
        {
            "Text (truncated)": (text[:80] + "…") if len(text) > 80 else text,
            "AI %": round(probabilities.get("ai", 0.0) * 100, 2),
            "Human %": round(probabilities.get("human", 0.0) * 100, 2),
            "Winner": predicted_label,
        }
    )
    st.session_state.history = history[-10:]  # keep the last 10 entries


def generate_feature_insights(features: Dict[str, float]) -> List[str]:
    """Return a few heuristic observations to pair with the raw stats."""
    insights: List[str] = []

    if features["burstiness"] < 0.2:
        insights.append("句長變異度低，句式較平均、節奏平穩。")
    elif features["burstiness"] > 0.6:
        insights.append("句長變異度高，句式長短落差大、較自由。")

    if features["lexical_diversity"] < 0.45:
        insights.append("詞彙多樣性較低，字詞重複度較高。")
    elif features["lexical_diversity"] > 0.65:
        insights.append("詞彙多樣性較高，字詞變化豐富。")

    if features["stopword_ratio"] < 0.35:
        insights.append("功能詞比例偏低，表達較精煉、直述。")
    elif features["stopword_ratio"] > 0.55:
        insights.append("功能詞比例偏高，語氣較口語、連接詞較多。")

    if features["punctuation_ratio"] < 0.03:
        insights.append("標點使用很少，句子多為簡短直述。")
    elif features["punctuation_ratio"] > 0.08:
        insights.append("標點使用較多，語氣或情緒轉折較頻繁。")

    if features["long_word_ratio"] > 0.35:
        insights.append("長詞比例偏高，帶有正式或技術說明感。")
    elif features["long_word_ratio"] < 0.22:
        insights.append("長詞比例偏低，較口語、隨筆的節奏。")

    if features["hapax_ratio"] > 0.6:
        insights.append("大量只出現一次的詞，表達較隨興、變化多。")
    elif features["hapax_ratio"] < 0.3:
        insights.append("重複詞較多，語言較公式化、重複性高。")

    if not insights:
        insights.append("特徵分佈接近中性，沒有明顯 AI/Human 偏向特徵。")

    return insights[:5]


def _set_sample_text(content: str) -> None:
    st.session_state.input_text = content


# Use samples drawn from the actual training set so the model's judgment aligns with buttons.
def pick_sample(kind: str) -> str:
    dataset = build_training_samples()
    pool = [text for label, text in dataset if label == kind]
    return random.choice(pool)


def main() -> None:
    st.title("AI vs Human 文章分類器")
    st.caption("輸入一段文本，按下按鈕後估計其 AI 生成或人類撰寫的機率。")
    st.info("使用流程：輸入或貼上文字 → 可用範例填充 → 按「開始判斷」 → 查看結果與特徵。")

    model, feature_names = train_model()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("輸入文本")
        input_text = st.text_area(
            "貼上或鍵入任意段落",
            value=st.session_state.get("input_text", ""),
            height=180,
            key="input_text",
        )

        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            st.button("填入 AI 範例", on_click=lambda: _set_sample_text(pick_sample("ai")))
        with sample_col2:
            st.button("填入 Human 範例", on_click=lambda: _set_sample_text(pick_sample("human")))
        st.caption("範例隨機取自模型的訓練樣本，方便快速試跑並與模型行為對齊。")

        analyze = st.button("開始判斷")

        st.markdown("---")

        result = st.session_state.get("last_result")
        if analyze and input_text.strip():
            probabilities = predict_text(model, feature_names, input_text)
            features = extract_features(input_text)
            add_to_history(input_text, probabilities)
            st.session_state["last_result"] = {
                "probabilities": probabilities,
                "features": features,
                "text": input_text,
            }
            result = st.session_state["last_result"]
        elif analyze and not input_text.strip():
            st.warning("請先輸入文字再按下開始判斷。")

        if result:
            probs = result["probabilities"]
            st.subheader("判斷結果")
            ai_pct = probs.get("ai", 0.0) * 100
            human_pct = probs.get("human", 0.0) * 100
            top_label = max(probs, key=probs.get) if probs else "-"
            col_a, col_b = st.columns(2)
            col_a.metric("AI 機率", f"{ai_pct:.1f}%")
            col_b.metric("Human 機率", f"{human_pct:.1f}%")
            render_probability_bar(probs)
            st.caption(f"最高機率：{top_label.upper()}（僅供參考，請搭配人工判斷）")
        else:
            st.info("輸入文字後按下「開始判斷」即可看到結果。")

    with col_right:
        st.subheader("輸入統計 / 特徵")
        result = st.session_state.get("last_result")
        if result:
            feature_map = result["features"]
            with st.expander("特徵表與解讀", expanded=True):
                render_features_table(feature_map)
                insights = generate_feature_insights(feature_map)
                st.markdown("**特徵觀察**")
                for note in insights:
                    st.markdown(f"- {note}")
        else:
            st.write("等待輸入中…")

        st.markdown("---")
        hist_title_col, hist_btn_col = st.columns([3, 1])
        with hist_title_col:
            st.subheader("最近推論")
        with hist_btn_col:
            if st.button("清空紀錄"):
                st.session_state["history"] = []
        history = st.session_state.get("history", [])
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True, height=240)
        else:
            st.caption("尚無紀錄，輸入文字後會自動顯示這裡。")

        st.markdown("---")
        st.subheader("模型說明")
        st.write(
            "此分類器使用小型樣本，基於文字特徵（詞長、多樣性、標點比例、句長變異、CJK 比例、hapax 等）"
            "訓練邏輯迴歸模型。結果僅供參考，建議搭配人工審核。"
        )


if __name__ == "__main__":
    main()
