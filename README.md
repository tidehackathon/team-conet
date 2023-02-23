# team-conet
CONET
## Release
All code and data is in the folder Release. Apart from usual imports we need:
* SentenceTransformer
* nltk
* pickle
* torch
* [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers), [documentation](https://simpletransformers.ai/)

All of these are open source and have excellent documentation. Excecuting the python programs should work if the two data files "annotated.tweets" and "EUDisinfo.txt" are in the same directory. Only the path to DisInformation-Challenge-Data needs to be adapted in ManipulationDetection.py.

### FactChecker.py
This program tests 37 AI modules for sentence similarity for the task of finding a statement in a set of (true or false) statements and chooses the best model. At the end the statements in EUDisinfo.txt are sorted by similarity to a couple of test statements. This shows how well a test statement can be matched semantically to the provided statements.

### ManipulationDetection.py
This program trains a RoBERTa model to detect manipulative formulations according to the data in annotated.tweets. It sorts 10000 Tweets of the provided data according to be manipulating or not within a few seconds on an RTX 2080.

## TIDE 2023
* [TIDE2023](https://tide.act.nato.int/mediawiki/tidepedia/index.php/2023_TIDE_Hackathon)
* [Team CONET](https://tide.act.nato.int/mediawiki/tidepedia/index.php/Team_1097)
* [Definition and references](https://tide.act.nato.int/mediawiki/tidepedia/index.php/Challenge_1038#tab=Definition)

## Data Sources of curated disinformation
* [EUvsDisinfo database](https://euvsdisinfo.eu/disinformation-cases/), [what is EUvsDisinfo](https://en.wikipedia.org/wiki/East_StratCom_Task_Force)
  * [20 entries](Datasets/EUDisinfo.txt)
  * [9076 prewar entries](Datasets/euvsdisinfo_v1_2.csv), until July 2020, tab separated, keywords (filter out coronavirus), summary, disproof. Can be used for database of correct information (disproof) and of disinformation (summary), [source](https://www.kaggle.com/datasets/imuhammad/euvsdisinfo-disinformation-database)
* [Kaggle Fake News](https://www.kaggle.com/datasets/mrisdal/fake-news)

## Manipulation, style, rhetorics
* [Identifying Disinformation Using Rhetorical Devices in Natural Language Models](https://www.osti.gov/biblio/1891194)
* [Fake News Classification with BERT](https://towardsdatascience.com/fake-news-classification-with-bert-afbeee601f41)

## Ideas
* build databases with true information and with false information
* compare twitter input semantically (AI, BERT etc.) with entries of both databases
* ignore most metadata, use date for matching twitter input to possibly related news entries
* classify twitter input in true/false/unknown depending on semantic similarities with entries in both databases
* Split articles from Guardian and NYT to get small pieces of true statements
  * split in paragraphs
  * remove paragraphs with less than 5 words (i.e. remove date and author)
* Clean twitter messages to get mainly messages that contain statements that can be analysed for disinformation
  * remove messages with links (main info is in the links which would need scraping)
