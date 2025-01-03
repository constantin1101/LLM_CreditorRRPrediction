# Creditor Recovery Rates Prediction with Earnings Calls and Large Language Models

This repository contains the research and implementation for the Master's Thesis:  
**"Creditor Recovery Rates Prediction with Earnings Calls and Large Language Models"**
**Karlsruhe Institute of Technology (KIT)**
**Constantin Hannes Ernstberger**

The goal of this project is to enhance the prediction of creditor recovery rates using advanced natural language processing (NLP) techniques applied to earnings calls. By combining textual data with financial and market indicators, the project aims to uncover valuable insights for investors and financial analysts.

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
  ├── anonymize_transcripts.ipynb         # Anonymize transcripts
  ├── extract_convert_calls.ipynb         # Processing: Extract transcripts from pdf files, save to csv
  ├── find_sections.ipynb                 # Summarize transcripts via OpenAI
├── combined_analysis.ipynb             # Use all extracted scores for bond-level RR prediction
├── composite_metrics.ipynb             # Build and us composite scores for more interpretable results
├── dealer_anaylsis.ipynb               # Trade-level RR prediction
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

---

## Future Work

---

## Contributing

---

## License