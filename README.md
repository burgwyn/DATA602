# DATA 602

## Getting Started

Install requirements

```shell
pip install -r requirements.txt
```

## Data Sets

All data is publicly avilable via [Recreational Information Database](https://ridb.recreation.gov)

- Reservations from 2021 - `FY21 Historical Reservations Full.csv`
- Facilities - `Facilties_API_v1.csv`

## Notebooks

- Reservations.ipynb
- ALS.ipynb
- SVD.ipynb
- Facilities.ipynb

## Data Cleaning

All data cleaning is in `Reservations.ipynb`.  The file loads the reservation data, cleans it and exports three CSVs for future analysis.

- REC.csv - cleaned reservation data
- REC_Collaborative_Facility.csv - dataset with user, item, and rating where item is facilityid - used in collaborative filtering
- REC_Collaborative_Product.csv - dataset with user, item, and rating where item is productid

## Collaborative Filtering

### ALS

The notebook requires PySpark, which can be run via Docker.

```shell
docker pull jupyter/pyspark-notebook:latest
docker run -p 8888:8888 jupyter/pyspark-notebook
```

Upload the notebook, `ALS.ipynb`, into the running Docker container to review and execute.

### SVD

SVD.ipynb - applies Singular Variable Decomposition

## Content Filtering

Facilities.ipynb loads Facility API data and creates models with TfdifVectorizer and CountVectorizer