# CovMythFact
A collection of 10000 news headlines related to covid-19, 5000 for fake news and true news each.

We make use of existing publised fake news datasets, e.g. COVID Fake News Dataset, CoAID, ReCOVery, and covid19-misinfo-data to construct CovMythFact for our linguistic analysis. We merge all the news headlines from the existing datasets, deduplicate the repeated headlines, and delete all the question titles (such as “How Long Does It Take for COVID-19 to Stop Being Contagious?”), keeping only statements in our final corpus. Finally, we obtain around 8,000 false headlines and 5,000 true headlines. In order to balance the two sub-corpora for a comparative study, we randomly sampled 5,000 false headlines from the 8,000 false headlines and finally make a balanced corpus (CovMythFact) containing 5,000 headlines for both true and false statements (10,000 headlines in total).

In addition to the headlines, we provide linguistic annotations to the data, including POS tagging, syntactic parsing, stylistic features, and conduct corpus-based enquiry for lexical semantic analysis, and many more linguistic features.


