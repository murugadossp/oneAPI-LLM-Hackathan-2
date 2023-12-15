from datetime import datetime
from pprint import pprint


def info(message):
    print(f"{datetime.now()}: {message}")


def pinfo(message):
    print(f"{datetime.now()}: data as below")
    pprint(message)
