import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bumblebee",
    version="0.5.0",
    author="Theodore Manassis & Alessia Tosi",
    author_email="author@example.com",
    description="A small nlp functions package",
    install_requires=["numpy","scipy" ,"pandas", "nltk", "textblob", "wordninja","scikit-learn","hypothesis"],
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


