# CS420 Content-Aware Video Summarization

## 👥 Group Members  

- Hồng Phúc Hải  
- Nguyễn Văn Giáp   

---

## Introduction

Video summarization aims to generate a short synopsis that captures the most informative and important parts of a video. This can be achieved in two main ways:

- **📷 Static Video Summary**: A collection of keyframes representing significant moments in the video.  
- **🎥 Dynamic Video Skimming**: A condensed version of the video containing key scenes.  

## Importance of Video Summarization

✅ **Saves Time**: Users can quickly understand the main content without watching the entire video.  
✅ **Efficient Video Browsing**: Helps users navigate large video collections efficiently.  

## Challenges

⚠️ **Diverse Contexts**: Different types of videos require different summarization techniques.  
⚠️ **Content Complexity**: Extracting meaningful information from complex video scenes.  
⚠️ **Maintaining Coherence**: Ensuring logical consistency in summarized content.  
⚠️ **Processing Time**: Computationally expensive models may not be efficient for real-time summarization.  

---

## Related Work

### 🎞️ **Unimodal Analysis (Visual-Only)**  

#### 🟢 **Unsupervised Learning**  
- **Adversarial Learning**: GANs extract key moments without supervision.  
- **Summary Properties**: Ensures diversity and representativeness.  

#### 🟢 **Supervised Learning**  
- **Temporal Dependency**: LSTMs/RNNs capture sequential dependencies.  
- **Spatio-Temporal Structure**: 3D CNNs/TCNs extract both spatial and temporal features.  
- **Adversarial Learning**: GAN-based methods for generating human-like summaries.  

#### 🟢 **Weakly-Supervised Learning**  
- **Key Motion Modeling**: Identifies crucial actions.  
- **Web Video Learning**: Utilizes similar online videos.  
- **Domain Adaptation**: Uses limited labels from related domains.  

### 📡 **Multimodal Analysis**  

#### 🟠 **Supervised Learning with Multimodal Inputs**  
- Incorporates **visual features** along with **textual metadata** (subtitles, transcripts) for more context-aware summaries.  

---

## Our Approach  

1️⃣ **Extract subtitles** using OpenAI **Whisper** for speech-to-text conversion.  
2️⃣ **Summarize subtitles** using an AI language model like **GPT**.  
3️⃣ **Match summarized content** with corresponding video segments.  
4️⃣ **Generate a final summarized video** combining selected video clips.  

---

## Evaluation  

### 📝 **Metrics for Text Summarization**  
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**  
  - **ROUGE-1**: Measures unigram (word-level) overlap.  
  - **ROUGE-2**: Measures bigram (two-word sequence) overlap.  
  - **ROUGE-L**: Measures longest common subsequence similarity.  

### 🎬 **Metrics for Video Summarization**  
- **F1-Score** (TVSum benchmark)  
- **Spearman’s Rank Correlation**  
- **Kendall’s Tau Correlation**  

---

## Results  

### 📑 **Text Summarization (English)**  

| Metric  | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| **ROUGE-1** | 0.9934 | 0.3972 | 0.5118 |
| **ROUGE-2** | 0.9498 | 0.3860 | 0.4944 |
| **ROUGE-L** | 0.9909 | 0.3964 | 0.5106 |

### 📑 **Text Summarization (Vietnamese)**  

| Metric  | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| **ROUGE-1** | 0.9928 | 0.1768 | 0.2942 |
| **ROUGE-2** | 0.9514 | 0.1710 | 0.2845 |
| **ROUGE-L** | 0.9792 | 0.1754 | 0.2918 |

### 🎥 **Video Summarization Results**  

| Metric | Score |
|--------|-------|
| **F1-Score** | 45.5 |
| **Spearman’s Rank Correlation** | 0.09 |
| **Kendall’s Tau Correlation** | 0.09 |

---

## Conclusion  

### ✅ **Advantages**  
✔️ Saves time and resources.  
✔️ Scalable and flexible.  

### ❌ **Disadvantages**  
⚠️ Visual information remains crucial for meaningful summaries.  
⚠️ Instability in summarization quality.  
⚠️ Struggles with non-dialogue segments.  

---

## Future Development  

🚀 **Improve handling of non-dialogue content.**  
🚀 **Enhance summarization coherence.**  
🚀 **Optimize processing time for real-time applications.**  

---

## 📩 Contact  

For any inquiries regarding this project, please reach out via email or open an issue on this GitHub repository.  
