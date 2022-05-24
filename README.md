# Camping Recommendation Systems for Recreation.gov

Nat Burgwyn
DATA 602
Spring 2022

## Getting Started

Install requirements

```shell
pip install -r requirements.txt
```

## Structure

```shell
├── ALS.ipynb
├── ALS.py
├── Facilities.ipynb
├── Hybrid.ipynb
├── README.md
├── Reservations.ipynb
├── SVD.ipynb
├── data
│   ├── REC.csv
│   ├── REC_Collaborative_Facility.csv
│   ├── REC_Collaborative_Product.csv
│   ├── REC_Content_Facility.csv
│   └── cs.csv
├── model
│   ├── als.model
│   └── svd.pkl
├── raw
│   ├── FY21\ Historical\ Reservations\ Full.csv
│   └── Facilities_API_v1.csv
└── requirements.txt
```

## Data Sets

All data is publicly avilable via [Recreational Information Database](https://ridb.recreation.gov) and kept in the `raw` directory.

- Reservations from 2021 - `FY21 Historical Reservations Full.csv`
- Facilities - `Facilties_API_v1.csv`

## Notebooks

- Reservations.ipynb
- ALS.ipynb
- SVD.ipynb
- Facilities.ipynb
- Hybrid.ipynb

## Data Cleaning

Data cleaning is in `Reservations.ipynb`.  The file loads the reservation data, cleans it and exports three CSVs into the `data` directory for future analysis.

- REC.csv - cleaned reservation data
- REC_Collaborative_Facility.csv - dataset with user, item, and rating where item is facilityid - used in collaborative filtering
- REC_Collaborative_Product.csv - dataset with user, item, and rating where item is productid

Content filtering data is cleaned in `Facilities.ipynb` and stored in `data`.

- cs.csv - cosine similarity scores from content filtering with CountVectorizer
- REC_Content_Facility.csv - clean facility data`

## Collaborative Filtering

### ALS

The notebook requires PySpark, which can be run via Docker.

```shell
docker pull jupyter/pyspark-notebook:latest
docker run -p 8888:8888 jupyter/pyspark-notebook
```

Upload the notebook, `ALS.ipynb`, into the running Docker container to review and execute.  It also requires `REC_Collaborative_Facility.csv` as an input. 

### SVD

`SVD.ipynb` applies Singular Value Decomposition using [Suprise](http://surpriselib.com/) library.  It requires `REC_Collaborative_Facility.csv` as an input.

## Content Filtering

`Facilities.ipynb` loads Facility API data and creates models with TfdifVectorizer and CountVectorizer

## Hybrid

`Hybrid.ipynb` loads CountVectorizer cosine similarity scores, `cs.csv`, and reference data `REC_Content_Facility.csv`.  In addition, it loads the ALS model to work in tandem with content filtering model to build hybrid recommendation function.  This notebook requires the Docker environment specified above.