import configparser
from utils.trainer import *

config = configparser.ConfigParser()
config.read('config.ini')

parser = Parser(config)
parser.parse()

evaluator = Evaluator(parser)
evaluator.evaluate()