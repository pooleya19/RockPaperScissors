# Run using the following command:
# py main.py

from DataCollector import DataCollector

if __name__ == "__main__":
    dataCollector = DataCollector()
    while dataCollector.running:
        dataCollector.update()