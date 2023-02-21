# team-conet
CONET
## TIDE 2023
* [TIDE2023](https://tide.act.nato.int/mediawiki/tidepedia/index.php/2023_TIDE_Hackathon)
* [Team CONET](https://tide.act.nato.int/mediawiki/tidepedia/index.php/Team_1097)

## Data Sources of curated disinformation
* [EUvsDisinfo database](https://euvsdisinfo.eu/disinformation-cases/)
  * [20 entries](Datasets/EUDisinfo.txt)
  * [9076 prewar entries](Datasets/euvsdisinfo_v1_2.csv), until July 2020, tab separated, keywords (filter out coronavirus), summary, disproof. Can be used for database of correct information (disproof) and of disinformation (summary), [source](https://www.kaggle.com/datasets/imuhammad/euvsdisinfo-disinformation-database)

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
