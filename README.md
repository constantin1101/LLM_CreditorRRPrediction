# Creditor Recovery Rates Prediction with Earnings Calls and Large Language Models

This repository contains the research and implementation for the Master's Thesis:  
**"Creditor Recovery Rates Prediction with Earnings Calls and Large Language Models"**
**Karlsruhe Institute of Technology (KIT)**
**Constantin Hannes Ernstberger**

Accurately forecasting recovery rates in distressed debt markets is vital for investors, analysts, and regulators. This study leverages large language models to mine earnings calls for insights on management tone, emotional cues, and analyst interactions. By combining these qualitative signals with established financial metrics, we create sophisticated in-sample composite measures at both the aggregate and trade level that capture transparency, awareness, and analysts’ support. Our approach yields improved out-of-sample predictions, underscoring the potential to enhance trading strategies and credit risk assessments. These findings highlight the critical role of earnings calls in shaping market confidence and recovery outcomes.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Datasets](#datasets)
7. [Methods](#methods)
8. [Results](#results)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

This project investigates:
- How management tone, sentiment, transparency and content as well as analysts reactions during earnings calls influence recovery rates.
- The impact of primary dealer intermediation on recovery outcomes.
- The effectiveness of Large Language Models (LLMs) in analyzing textual financial data compared to traditional NLP approaches.

By leveraging features such as sentiment, emotional cues, and structured content, the model achieves better prediction accuracy than existing methods.

---

## Key Features

- **Textual Feature Analysis**:
  - Extracted from earnings calls' presentation and Q&A sections.
  - Analyzed using LLMs and the GoEmotions model to identify 27 emotions.

- **Composite Metrics**:
  - **Management Transparency Score (MTS)**: Assesses management clarity and responsiveness.
  - **Crisis Awareness and Management Index (CAMI)**: Measures management's handling of crises and financial health.
  - **Analysts Urgency Score (AUS)**: Tracks analyst concerns and urgency in the Q&A.

- **Predictive Models**:
  - Combines bond pricing, transaction data, and earnings call features.
  - Incorporates dimensionality reduction techniques like PCA for large feature sets.

---

## Project Structure

```plaintext
├── data/                               # Aggregated bond recovery rate data
├── dealer/                             # Trade-level bond recovery rate data
├── goemotions/                         # Emotion scores
├── models/                             # Saved models
├── prompts/                            # Prompts to extract scores from earnings calls and summarize them
├── CoT/                                # Prompts for Chain-of-Thought approach
├── transcripts/                        # Earnings calls transcripts with and without scores
├── results/                            # Saved model outputs and evaluation metrics
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── preprocessing/                      # Data preprocessing steps
  ├── aggregate_transcripts.ipynb         # Aggregate earnings call based on call ID
  ├── extract_convert_calls.ipynb         # Processing: Extract transcripts from pdf files, save to csv
  ├── find_sections.ipynb                 # Summarize transcripts via OpenAI
├── combined_analysis.ipynb             # Use all extracted scores for bond-level RR prediction
├── composite_metrics.ipynb             # Build and us composite scores for more interpretable results
├── dealer_anaylsis.ipynb               # Trade-level RR prediction and primary-dealer classification
├── dealer_exploration.ipynb            # Trade-level and dealer intermediation data exploration
├── goemotions.ipynb                    # Compute emotion scores via GoEmotions LLM from Google
├── presentation_analysis.ipynb         # Extract scores from presentation via OpenAI
├── qna_analysis.ipynb                  # Extract scores from Q&A via OpenAI
├── cot_analysis.ipynb                  # Chain-of-Thought as potential future research approach

---

## Usage

---

## Datasets

Included Data
- Processed Earnings Call Data: Preprocessed textual data extracted from earnings call transcripts.
- Extracted Features: Sentiment, emotions, and structured content-specific features derived from the textual data.
Source Data (Not Included)
- Raw Earnings Calls: Obtainable from financial data providers (e.g., S&P Capital IQ, Refinitiv).
- Bond Data: Pricing, transaction history, and recovery rate data from bond markets.
Emotion Data
- Emotion scores are generated using the GoEmotions model, which classifies text into 27 emotion categories for granular sentiment analysis.

---

## Methods

Textual Feature Extraction
Earnings Calls:
- Presentation Section: Analyzing management's narrative for sentiment and transparency.
- Q&A Section: Extracting analyst concerns and management responsiveness.
- Emotion Analysis: Leveraging GoEmotions to classify text into 27 distinct emotions.
Composite Metrics
- Management Transparency Score (MTS): Evaluates clarity, vagueness, and responsiveness.
- Crisis Awareness and Management Index (CAMI): Measures confidence and financial situational awareness.
- Analysts Urgency Score (AUS): Indicates urgency based on analyst satisfaction and concern.
Modeling Techniques
- LLM-based Analysis: Utilizing fine-tuned models for sentiment and emotion classification.
- Feature Engineering: Employing dimensionality reduction (e.g., PCA) to optimize large feature sets.
- Predictive Models: Combining textual features with bond pricing and transaction data for recovery rate prediction.

---

## Results

├── Large Language Models for Creditor Recovery Rates.pdf

Earnings calls are key information events that provide rich qualitative insights into corporate performance and financial health. Our study highlights their critical role in driving market behavior and underscores the importance of earnings calls in reshaping market expectations-particularly in the distressed debt market. The increased volatility in recovery rates immediately following earnings calls reflects the rapid incorporation of new information into investor pricing and decision making.

Motivated by these observations, we used large language models to integrate textual and emotional features from earnings calls into traditional credit risk models. By capturing latent dimensions of communication-such as management tone, transparency, and emotional clarity-we demonstrated that these qualitative signals improve recovery rate predictions. The inclusion of textual features led to improved predictive metrics, particularly at longer horizons, confirming the economic value of earnings calls as a timely source of actionable information. Models incorporating rich linguistic features consistently outperformed those relying solely on financial data.

Our analysis revealed different responses among market participants. Primary dealers were more influenced by transparent and precise communication, while non-primary dealers responded to signals of optimism and empathy. This heterogeneity highlights the need for tailored communication strategies in distressed debt markets to address different stakeholder priorities. The observed trading patterns underscore the importance of earnings calls not only as catalysts for short-term market activity, but also as drivers of longer-term recovery trajectories. Their post-call stabilization of recovery rates suggests that earnings calls serve as key events for market recalibration, influencing liquidity, price stability, and recovery outcomes.

Crucially, we also demonstrate the potential success of these integrated composite metrics in trading applications. Simulated strategies using both textual and emotional cues alongside traditional financial measures showed improved risk-adjusted returns and outperformance relative to models based solely on conventional data. These results underscore the broader economic significance of earnings calls in bond markets and highlight the practical utility of enhanced language-based analytics.

In addition to improving predictive accuracy, this study has practical implications for investors, analysts, and regulators. By systematically linking narrative cues to recovery outcomes, we demonstrate how LLM-driven approaches can uncover early warning signals and enable more informed decision-making in distressed credit environments. Looking forward, extending this framework to incorporate larger datasets and further innovations in natural language processing offers promising potential for navigating the complexities of financial markets and responding effectively to periods of heightened uncertainty.

---

## Future Work

Future research can extend the findings of this study by addressing its limitations and exploring new methodologies to improve predictive accuracy and interpretability. Expanding the dataset to include earnings calls from international markets and a broader range of firms, including those not in financial distress, would provide a more comprehensive understanding of recovery rates. Including historical earnings calls from different economic cycles could reveal how management tone, content, and strategy evolve over time, providing insights into the dynamics of recovery rates in different market conditions.

Advances in natural language processing techniques, particularly through the use of interpretable approaches such as chain-of-thought prompting and more sophisticated reasoning frameworks, offer significant potential. These techniques can increase the robustness and granularity of textual feature extraction, provide clearer explanations of model predictions, and enable direct applicability to decision makers. Multimodal analysis - integrating textual, numerical, and potentially even visual data - could further enrich models by capturing a wider range of signals that influence recovery outcomes.

Research on the role of explainability and interpretability in credit risk modeling should also be prioritized. Building on insights from explainable machine learning studies \citep{bell2024glassbox}, future work could develop transparent recovery rate models that reveal meaningful interactions between bond characteristics, macroeconomic indicators, and qualitative features. These models would not only improve predictive power, but also foster greater confidence among practitioners.

Finally, studying the long-term impact of trader behavior on credit market dynamics could provide critical insights into the interplay between qualitative communication and market outcomes. Understanding how trading patterns and intermediation affect recovery rates and market stability over longer periods would deepen knowledge of the mechanisms underlying credit risk and inform more effective policy interventions. By addressing these areas, future research can advance the integration of qualitative and quantitative methods to ensure more accurate, interpretable, and actionable insights into recovery rate forecasting.

---

## License

Constantin Hannes Ernstberger
constantin.e99@web.de