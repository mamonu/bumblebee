import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlpfunctions",
    version="0.0.1",
    author="Theodore Manassis & Alessia Tosi",
    author_email="author@example.com",
    description="A small nlp functions package",
    install_requires=["numpy", "pandas", "nltk"],
    data_files=[('textfile', ['nlpfunctions/words-by-freq125K.txt'])],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mamonu/textconsultations",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)


