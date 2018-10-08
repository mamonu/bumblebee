# bumblebee
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)
![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
![travis](https://travis-ci.com/mamonu/bumblebee.svg?branch=master)
![nlp](https://github.com/mamonu/bumblebee/blob/master/pics/subject-NLP-lightgrey.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

<img src="https://github.com/mamonu/bumblebee/blob/master/pics/bb.png" align="right"
     title="nlp logo" width="197" height="170">
     
<br>


useful nlp functions / pipelines / transformers for text mining in a package format


<br>

<br>


## Getting Started


TODO: add when ready

### structure:

<img src="https://github.com/mamonu/bumblebee/blob/master/pics/bumblebeediagram.png" align="center"
     title="structure" >


### Prerequisites


This package is compatible with Linux/OSX systems


#### What package prerequsites you need to install the software ? 

See [requirements.txt](https://github.com/mamonu/bumblebee/blob/master/requirements.txt) 


#### How to install those prerequisite packages if you need to?

Most prerequisite packages will be installed automaticaly.

However the spellcheck submodule uses the enchant system library.If this library is not there then pyenchant will not work.

On an Ubuntu / Debian system that means that you should run on your bash shell:   `sudo apt-get install enchant`

On a RedHat / CentOS / Cloudera CDH system that means you should run on your bash shell:   `sudo yum install enchant`

On OSX you need to run `brew install enchant`


After that the fastest way to make sure you have everything would be go to main directory and run 

```
pip install -r requirements.txt 
python -m nltk.downloader vader_lexicon stopwords wordnet brown_tei gutenberg punkt popular
```

after downloading these packages and their assorted material (in the case of NLTK) everything should run smoothly.
If not please open an issue here on this repo.



### Installing

TODO: add when ready    (at some point its going to be `pip install nplfunctions` )

## Running the tests

((you will need pytest intalled. if you dont have it the just  :  `pip install pytest pytest-cov`    )

In order to run all the automated tests for this system  after you have cloned it into your system just do:

```
cd tests

pytest -v   ## run all tests

cd ..

pytest -v --cov=nlpfunctions tests/      ## run tests and calculate testing coverage 


```


## Continuous Integration

Continuous Integration is a software development practice where members of a team 
integrate their work on a main repo frequently. Usually each person integrates their work at least daily
leading to multiple integrations per day. Each integration is verified by an automated build 
(that includes running an automated test harness) to detect integration errors as quickly as possible. 
Many teams find that this approach leads to significantly reduced integration problems 
and allows a team to develop cohesive software more rapidly.

<br>

We are using Travis CI <img src="https://github.com/mamonu/bumblebee/blob/master/pics/travis.png" title="travis logo" width="50" height="50">       for this process. 



<br>

<br>


## Built With

<img src="https://github.com/mamonu/bumblebee/blob/master/pics/blacklogo2.png" align="right" title="black logo" width="56" height="27"><img src="https://github.com/mamonu/bumblebee/blob/master/pics/sphinximage.png" align="right" title="sphinx logo" width="40" height="34">
<img src="https://github.com/mamonu/bumblebee/blob/master/pics/NLTK.png" align="right" title="nltk logo" width="50" height="50"> <img src="https://github.com/mamonu/bumblebee/blob/master/pics/scikit-learn-logo-small.png" align="right" title="sklearn logo"> <img src="https://docs.pytest.org/en/latest/_static/pytest1.png" align="right" title="pytest logo" width="50" height="50"><img src="https://github.com/mamonu/bumblebee/blob/master/pics/travis.png" align="right" title="travis logo" width="50" height="50">





* [NLTK](https://github.com/nltk/nltk) - Natural Language ToolKit
* [scikit-learn](http://scikit-learn.org/) - machine learning framework 
* [pytest](https://docs.pytest.org/en/latest/) - unit testing framework
* [black](https://github.com/ambv/black) - code formatter
* [sphinx-doc](http://www.sphinx-doc.org/en/master/) - documentation framework
* [Travis CI](https://travis-ci.org/) - Continuous Integration framework


## Contributing

Please read [CONTRIBUTING.md](https://github.com/mamonu/bumblebee/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.



## Authors / Maintainers

* **Theodore Manassis**  - [mamonu](https://github.com/mamonu)
* **Alessia Tosi** - [exfalsoquodlibet](https://github.com/exfalsoquodlibet)


See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/mamonu/bumblebee/blob/master/LICENCE.md) file for details

## Acknowledgments

    In opensource everyone is standing on the shoulders of giants... 
    or possibly a really tall stack of ordinary-height people


The authors would like to thank in no particular order:

- the ONS Big Data team (check their repos [here](https://github.com/ONSBigData)    )

- the NLTK maintainers

- the scikit-learn maintainers

- [Benjamin Bengfort](https://github.com/bbengfort), [Tony Ojeda](https://github.com/ojedatony1616), [Rebecca Bilbro](https://github.com/rebeccabilbro) . The authors of one of the most useful NLP books out there:      
                  [Applied Text Analysis with Python](http://shop.oreilly.com/product/0636920052555.do)



---


## references

- Blei, David M.; Ng, Andrew Y.; Jordan, Michael I (January 2003). Lafferty, John, ed. "Latent Dirichlet Allocation". Journal of Machine Learning Research. 3 (4–5): pp. 993–1022.

- Blei, David (April 2012). "Probabilistic Topic Models". Communications of the ACM. 55 (4): 77–84. 

- Lee, Daniel D., and H. Sebastian Seung. "Learning the parts of objects by non-negative matrix factorization." Nature 401.6755 (1999): 788.

-  Bird, S., Klein, E., & Loper, E. (2009). Natural language processing with Python: analyzing text with the natural language toolkit. " O'Reilly Media, Inc.".

- Mihalcea, R., & Tarau, P. (2004). Textrank: Bringing order into text. In Proceedings of the 2004 conference on empirical methods in natural language processing.

- Li, W. (1992). Random texts exhibit Zipf's-law-like word frequency distribution. IEEE Transactions on information theory, 38(6), 1842-1845.

- Knuth, D. E., Morris, Jr, J. H., & Pratt, V. R. (1977), 'Fast pattern matching in strings'. SIAM journal on computing, 6(2), 323-350.

- Gilbert, C. H. E. (2014). Vader: A parsimonious rule-based model for sentiment analysis of social media text. In Eighth International Conference on Weblogs and Social Media (ICWSM-14). Available at (08/10/18) http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf






