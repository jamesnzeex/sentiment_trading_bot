# Sentiment Trading Bot

This repository contains a simple python implementation of sentiment trading bot using MLP for sentiment analysis on fianancial news to trigger buy/sell signal for a particular financial instrument.

The bot is segmented into 4 sections
- Webscrapping of News from [FINVIZ](https://finviz.com/)
- Pretraining of MLP model for sentiment classification
  - Dataset is obtained from [Kaggle](https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news)
  - Note: classification accuracy is approximately 70%, using more complex state of the art NLP model would improve the accuracy
- API interface between [trading platform](https://alpaca.markets/) and bot
- Trading strategy - simple trading strategy based on sentiment voting

## Future Work
- [ ] To use finBERT model for sentiment classification
- [ ] To use other technical indicator for trading
- [ ] To include other sources (Twitter/Reddit) for sentiment analysis on financial market
- [ ] To validate trading strategy on paper trading account
