from fr_utils import *
from inception_network import *

from helpers import *

name = input('enter your name to recognise')
print('saving name to database')
add_to_database(name)