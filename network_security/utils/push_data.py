import os
import sys
import json
from dotenv import load_dotenv
import certifi
import pandas as pd
import pymongo
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

load_dotenv()

mongodb_url = os.getenv("MONGODB_URL_KEY")
ca = certifi.where()


class NetworkDataExtractor:
    def __init__(self, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.client = pymongo.MongoClient(mongodb_url, tlsCAFile=ca)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_converter(self, csv_file_path: str):
        try:
            data = pd.read_csv(csv_file_path)
            data.reset_index(drop=True, inplace=True)

            records = list(json.loads(data.T.to_json()).values())

            return records

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def push_data_to_mongodb(self, data: list):
        try:
            db = self.client[self.database]
            collection = db[self.collection]
            collection.insert_many(data)

            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    FILE_PATH = "network_data/phisingData.csv"
    DATABASE = "NetworkSecurity"
    COLLECTION = "PhishingData"

    network_obj = NetworkDataExtractor(database=DATABASE, collection=COLLECTION)
    data = network_obj.csv_to_json_converter(csv_file_path=FILE_PATH)
    status = network_obj.push_data_to_mongodb(data=data)
    if status:
        logging.info("Data pushed to MongoDB successfully.")
    else:
        logging.error("Failed to push data to MongoDB.")
