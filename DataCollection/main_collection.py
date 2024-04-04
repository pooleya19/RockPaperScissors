# Run using the following command:
# py main_collection.py

from DataCollector import DataCollector

dataCollector = DataCollector()
while dataCollector.running:
    dataCollector.update()