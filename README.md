# Voice of the Customer: Sarcasm-Aware Sentiment Analysis of Swiggy App Reviews

> A text-analytics pipeline that scrapes Google Play reviews of the Swiggy app, measures customer sentiment with three competing models, corrects for sarcasm, finds out *why* customers are unhappy, and turns the findings into a marketing playbook with KPIs.

**Course:** Text Analytics · End-Term Project
**Institute:** Prin. L. N. Welingkar Institute of Management Development & Research
**Team:** _[Add names here]_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/Swiggy_Sentiment_Analysis.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of contents

1. [The marketing problem](#the-marketing-problem)
2. [Key findings](#key-findings)
3. [Pipeline](#pipeline)
4. [Results in charts](#results-in-charts)
5. [How to run](#how-to-run)
6. [Repository structure](#repository-structure)
7. [Tech stack](#tech-stack)
8. [Limitations and future scope](#limitations-and-future-scope)
9. [Ethics and data use](#ethics-and-data-use)

---

## The marketing problem

Swiggy competes with Zomato (food delivery) and Zepto / Blinkit (quick commerce). Switching costs are close to zero, so every bad experience can cost a customer, and the Play Store rating is the first thing a new user sees. Management needs to know:

1. How do customers feel about Swiggy right now?
2. **What drives** that feeling: delivery time, fees, refunds, the app?
3. How much negative feeling is **hidden behind sarcasm** (*"Great job Swiggy, food came cold after 2 hours!"*), which basic sentiment tools read as praise?
4. What should the marketing and customer-experience teams **do** about it?

---

## Key findings

Based on the 3,000 newest English reviews (23–30 Sep 2026), of which **1,145** remained after cleaning.

| Metric | Value |
|---|---|
| Sentiment split (sarcasm-adjusted) | 61% negative · 9% neutral · 30% positive |
| Net Sentiment Score | **−31** |
| Average rating of analysed reviews | 2.45★ (Play Store overall: 4.5★) |
| Best sentiment model | **RoBERTa** (accuracy 86.2%, macro-F1 0.67) |
| Sarcastic reviews detected | 24 (2.1% of all, 3.4% of 1–2★ reviews) |
| Happy-vs-unhappy classifier (TF-IDF + Logistic Regression) | 90.7% accuracy, macro-F1 0.90 (5-fold CV) |

**Model comparison against the customers' own star ratings**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) | **0.862** | **0.671** |
| VADER | 0.741 | 0.570 |
| TextBlob | 0.667 | 0.523 |

**Top pain points** (pain score = mention share × % negative)

| Rank | Aspect | Mentioned in | % negative | Pain score |
|---|---|---|---|---|
| 1 | Refund & customer support | 21.6% of reviews | 97.2% | 21.0 |
| 2 | Delivery time | 20.7% | 78.5% | 16.2 |
| 3 | Pricing & fees | 15.5% | 78.7% | 12.2 |

**What this means**

- **Written feedback is far more negative than the star rating suggests.** The short reviews removed during cleaning ("good", "nice app") averaged 4.2★. Happy customers rate and leave; unhappy customers explain. The 4.5★ store rating hides the tone of detailed feedback.
- **Support and refunds are the biggest problem**, not delivery speed. Almost every review that mentions them is negative.
- **Fee complaints carry the most public weight.** Pricing & fees reviews collected 239 "helpful" votes, the most of any aspect, so they are what prospective users read first.
- **Sarcasm is small in volume but concentrated.** It is most common in complaints about order accuracy and food quality, where customers are most exasperated.

The full playbook with actions and KPIs is in Section 11 of the notebook and in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Pipeline

```
 1 Collect  →  2 Clean  →  3 Score  →  4 Sarcasm  →  5 Features  →  6 Act
```

| # | Step | What happens | Main tools |
|---|---|---|---|
| 1 | **Collect** | Scrape the 3,000 newest English Play Store reviews + star ratings | `google-play-scraper` |
| 2 | **Clean** | Drop duplicates, reviews under 3 words and Hindi/Hinglish; normalise slang; build two text versions (light for sentiment, full for n-grams) | `pandas`, `re`, `nltk`, `contractions` |
| 3 | **Score** | VADER vs TextBlob vs RoBERTa, each tested against the star ratings | `vaderSentiment`, `textblob`, `transformers` |
| 4 | **Sarcasm** | Hybrid detector: rating–text mismatch, cue phrases, irony transformer | `re`, `transformers` |
| 5 | **Features** | TF-IDF, Logistic Regression, NMF topics, tri-grams, unigrams, word clouds | `scikit-learn`, `wordcloud` |
| 6 | **Act** | Aspect pain scores → ranked playbook with KPIs | `pandas` |

The reasoning behind every design decision is explained in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Results in charts

**Star ratings and review length.** Unhappy customers write about four times more than happy ones.
![EDA](images/fig01_eda_ratings_and_length.png)

**Sentiment model comparison**
![Model comparison](images/fig02_sentiment_model_comparison.png)

**Model score by star rating** (sanity check: scores should rise with the stars)
![Score by rating](images/fig03_sentiment_score_by_rating.png)

**TF-IDF features**
![TF-IDF](images/fig04_tfidf_features.png)

**Tri-grams: negative vs positive reviews**
![Trigrams](images/fig05_trigrams.png)

**Unigrams**
![Unigrams](images/fig06_unigrams.png)

**Word clouds: all, negative and positive reviews**
![Word clouds](images/fig07_wordclouds.png)

**Aspect-based sentiment and pain scores**
![Aspects](images/fig08_aspect_sentiment.png)

**Share of negative reviews over time**
![Trend](images/fig09_negative_trend.png)

---

## How to run

### Option A: Google Colab (easiest)

Click the **Open in Colab** badge above, then choose **Runtime → Run all**. The notebook installs any missing library itself. A GPU runtime makes the two RoBERTa models run in seconds instead of about 3 minutes each.

### Option B: Locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Swiggy_Sentiment_Analysis.ipynb
```

### Configuration

All settings are in one block at the top of the notebook:

| Setting | Default | Meaning |
|---|---|---|
| `APP_ID` | `in.swiggy.android` | Change this to analyse another app (e.g. Zomato: `com.application.zomato`) |
| `N_REVIEWS` | `3000` | Number of newest reviews to scrape |
| `MIN_WORDS` | `3` | Reviews shorter than this are dropped |
| `ENGLISH_THRESHOLD` | `0.60` | Share of recognised English words needed to keep a review |
| `USE_TRANSFORMERS` | `True` | Set `False` to skip RoBERTa (faster, no PyTorch needed) |
| `REFRESH_DATA` | `False` | `True` scrapes fresh reviews even if the CSV already exists |

> **Note:** results will differ from the numbers above when you re-run, because the scraper always fetches the *newest* reviews.

### Outputs

Everything is written to `swiggy_project_outputs/`:

| File | Contents |
|---|---|
| `01_swiggy_reviews_raw.csv` | Scraped reviews (user names and photos removed) |
| `02_swiggy_reviews_clean.csv` | Cleaned reviews with both text versions |
| `03_sarcastic_reviews.csv` | Reviews flagged as sarcastic and which signal fired |
| `04_trigrams.csv`, `05_unigrams.csv` | N-gram frequency tables |
| `06_aspect_sentiment.csv` | Aspect-level sentiment and pain scores |
| `07_insights_and_strategy_report.txt` | Auto-generated findings and strategy |
| `08_swiggy_reviews_scored.csv` | Final dataset with every model score |
| `fig01`–`fig09` `.png` | All charts |

---

## Repository structure

```
.
├── Swiggy_Sentiment_Analysis.ipynb   # Main notebook (code, outputs, explanations)
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT licence
├── .gitignore
├── docs/
│   └── METHODOLOGY.md                # Logic and reasoning behind every step
├── data/
│   └── README.md                     # Data source, data dictionary, privacy notes
└── images/                           # Charts used in this README
```

---

## Tech stack

| Purpose | Libraries |
|---|---|
| Data collection | `google-play-scraper` |
| Data handling | `pandas`, `numpy` |
| Text cleaning | `re`, `nltk` (stopwords, WordNet lemmatiser, POS tagger), `contractions` |
| Sentiment | `vaderSentiment`, `textblob`, `transformers` + `torch` (RoBERTa) |
| Sarcasm | `transformers` (RoBERTa irony model), regex cue phrases |
| Features and models | `scikit-learn` (TF-IDF, CountVectorizer, Logistic Regression, NMF, metrics) |
| Visualisation | `matplotlib`, `wordcloud` |

---

## Limitations and future scope

- **Sample:** the 3,000 newest reviews cover about one week. They describe the *current* experience, not a long-term trend. Re-running monthly would build that trend.
- **Language:** Hindi and Hinglish reviews were excluded. A multilingual model such as XLM-RoBERTa or MuRIL could include them.
- **Aspects:** tagging is keyword-based. An aspect-based sentiment model could score each aspect separately inside a mixed review.
- **Sarcasm:** labels are not hand-verified. Labelling a sample of about 200 reviews would measure the detector's precision and recall. Two of the three sarcasm signals use the star rating, so part of the accuracy gain is by design.
- **Self-selection:** people who write reviews skew towards strong opinions, so the results describe vocal customers rather than all customers.
- **Neutral class:** with only 47 three-star reviews, the neutral class is hard for every model (F1 0.22), which pulls macro-F1 down.

---

## Ethics and data use

- Only publicly visible reviews were collected, at a polite rate (1-second pause between requests).
- User names and profile pictures are dropped immediately after scraping; they are personal data the analysis does not need.
- The scraped CSVs are **not** committed to this repository (see `.gitignore`). Run the notebook to regenerate them.
- This is an academic project. It is not affiliated with or endorsed by Swiggy or Google.

---

## License

Released under the [MIT License](LICENSE).
