# Description of the dataset

The dataset is composed of two files:

- `en_docs_clean.csv`
- `pt_docs_clean.csv`

The first contains manifestos from the United Kingdom. The second contains
manifestos from Portugal. You can use one of them or both in your system.

Each **line** in a file contains the following columns:

- **text**: one segment of a manifesto
- **manifesto_id**: the id for the manifesto to which the segment belongs
- **party**: the political party that authored the manifesto
- **date**: the election date (year and month)
- **title**: the title of the manifesto

Columns are separated by commas (,). Strings that contain commas are delimited
by double quotes ("). Both files are encoded using UTF-8.

#Ex 1 guide
#running normally
cd Ex1
python3 Ex1.py

#libraries used
- `pandas`
- `whoosh`
- `nltk`

#command flags
- `raw_data` - runs using raw dataset instead of cleaned dataset 
- `lab`  - runs algorithm based on the version made at laboratory 2 in the course
- `no_generate`- use this if indexdir is already created from previous execution, and you did't change any flag. just for time saving on indexing the files.

#examples

```python3 Ex1.py ``` - runs normally on cleaned dataset (en_docs_clean.csv) and indexes the files

```python3 Ex1.py lab ```- runs normally on cleaned dataset (en_docs_clean.csv) and indexes the files in memory using lab method

```python3 Ex1.py raw_data ``` - runs on the uncleaned dataset (en_docs.csv) and indexes the files

```python3 Ex1.py raw_data lab ```- runs on the uncleaned dataset (en_docs.csv) and indexes the files in memory using lab method

