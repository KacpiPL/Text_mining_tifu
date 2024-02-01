# Reddit TIFU

## Project Overview

The project, centered around the Reddit TIFU (Today I F**ked Up) subreddit, applies various text minig approach and natural language processing (NLP) techniques, primarily clustering and classification. In case of cleaning the text we start with regexes, stop words removal, and compare stemming and lemmatization. 

## Objective

The objective of the Reddit TIFU DataFrame project in text mining is to comprehensively analyze and understand the patterns, themes, and sentiment inherent in the personal stories shared on the TIFU subreddit.

## Data Source

Our analysis is fueled by data sourced from [huggingface.co]([http://insideairbnb.com/](https://huggingface.co/datasets/reddit_tifu?row=77)). We used "datasets" library to import this dataset directly from the python.

When it comes to the classification problem we scraped the thread TIFU using [apify.com](https://apify.com/trudax/reddit-scraper) scraper, which is available for free for one day. This way we were able to get the over18 column and perform classification on it.

## Data frame

**Hugging face df**
* ups - upvotes
* num_comments - number of comments
* upvote_ratio - upvotes ratio
* score - score
* documents - post text without tldr
* tldr - tldr line
* title - trimmed title without tldr

However we mainly use documnents column.

**Scraped df**
* body - post text without tldr
* numberOfcomments - number of comments
* over18 - T/F flag
* upVotes - upvotes
