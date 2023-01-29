import os
import tarfile
from urllib.request import urlretrieve

DOWNLOAD_ROOT = "https://github.com/alexhegit/handson-ml2/raw/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # if not os.path.isdir(housing_path):
    #     os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # print("tgz_path:" + tgz_path)
    #
    # urlretrieve(housing_url, tgz_path)
    #
    # print("housing_url:" + housing_url)

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data(HOUSING_URL, HOUSING_PATH)