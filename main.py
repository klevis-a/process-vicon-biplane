import os
import psutil
import time
import sys
from misc.file_utils import pretty_size
from database import create_db
from parameters import Params


process = psutil.Process(os.getpid())
print(pretty_size(process.memory_info().rss))

start = time.time()
params = Params.get_params(sys.argv[1])
db_dir = params.db_dir
db, anthro = create_db(db_dir)
end = time.time()
print(end-start)

print(pretty_size(process.memory_info().rss))
