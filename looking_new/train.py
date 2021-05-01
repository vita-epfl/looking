import configparser
from utils.trainer import *

config = configparser.ConfigParser()
config.read('config.ini')

parser = Parser(config)
parser.parse()

trainer = Trainer(parser)
trainer.train()