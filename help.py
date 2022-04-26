import click
import pandas as pd
from dotenv import load_dotenv
import os
import logging
from tabulate import tabulate

logging.basicConfig() 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER_PATH")
MODELS_CSV = os.path.join(DATA_FOLDER, "models.csv")

@click.command()
@click.option('-l', '--list', is_flag=True, default=False, help='List available models.')
def help(list):
  if list:
    models = pd.read_csv(MODELS_CSV)
    pd.options.display.max_columns = len(models.columns)
    print(tabulate(models, headers="keys", tablefmt="fancy_grid"))
  else:
      print("""
          Usage: help.py [OPTIONS]
          Try 'help.py --help' for help.
      """)

if __name__ == "__main__":
    help()