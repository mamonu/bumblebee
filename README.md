# nlpfunctions
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)
![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
![Conda](https://img.shields.io/conda/pn/conda-forge/python.svg)
![nlp](https://github.com/mamonu/textconsultations/blob/master/pics/subject-NLP-lightgrey.svg)

<img src="https://github.com/mamonu/textconsultations/blob/master/pics/glitchynlp.gif" align="right"
     title="nlp logo" width="178" height="178">
     
<br>


useful nlp functions for text mining in a package format


<br>

<br>


## Getting Started



### Prerequisites

What package prerequsites you need to install the software ? See [requirements.txt](https://github.com/mamonu/textconsultations/blob/master/requirements.txt) 


How to install those prerequisite packages if you need to?

go to main directory and run 

```
pip install -r requirements.txt 
python -m nltk.downloader all

```

after downloading these packages and their assorted material (in the case of NLTK) everything should run smoothly 



### Installing



## Running the tests

In order to run all the automated tests for this system after you have cloned it into your system just do:

```
cd tests

pytest -v

```


## Built With
<img src="https://github.com/mamonu/textconsultations/blob/master/pics/blacklogo2.png" align="right" title="black logo" width="56" height="27"><img src="https://github.com/mamonu/textconsultations/blob/master/pics/sphinximage.png" align="right" title="sphinx logo" width="40" height="34">
<img src="https://github.com/mamonu/textconsultations/blob/master/pics/NLTK.png" align="right" title="nltk logo" width="50" height="50"> <img src="https://github.com/mamonu/textconsultations/blob/master/pics/scikit-learn-logo-small.png" align="right" title="sklearn logo"> <img src="https://docs.pytest.org/en/latest/_static/pytest1.png" align="right" title="pytest logo" width="50" height="50">




* [NLTK](https://github.com/nltk/nltk) - Natural Language ToolKit
* [scikit-learn](http://scikit-learn.org/) - machine learning framework 
* [pytest](https://docs.pytest.org/en/latest/) - unit testing framework
* [black](https://github.com/ambv/black) - code formatter
* [sphinx-doc](http://www.sphinx-doc.org/en/master/) - documentation framework


## Contributing

Please read [CONTRIBUTING.md](https://github.com/mamonu/textconsultations/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.



## Authors / Maintainers

* **Theodore Manassis**  - [mamonu](https://github.com/mamonu)
* **Alessia Tosi** - [exfalsoquodlibet](https://github.com/exfalsoquodlibet)


See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/mamonu/textconsultations/blob/master/LICENCE.md) file for details

## Acknowledgments

    In opensource everyone is standing on the shoulders of giants... 
    or possibly a really tall stack of ordinary-height people


The authors would like to thank in no particular order:

- the Big Data team (check their repos [here](https://github.com/ONSBigData)    )

- the NLTK people

- the scikit-learn people

- [Benjamin Bengfort](https://github.com/bbengfort), [Tony Ojeda](https://github.com/ojedatony1616), [Rebecca Bilbro](https://github.com/rebeccabilbro) . The authors of one of the most useful NLP books out there: 
        [Applied Text Analysis with Python](http://shop.oreilly.com/product/0636920052555.do)



---




### structure:




├── nlpfunctions

│   ├── basic_NLP_functions

│   ├── similarities

│   ├── spellcheck

│   ├── textranksummary

│   ├── TopicMod_functions

│   ├── utils
