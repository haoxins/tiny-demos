import os

import lance

cur_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(cur_dir, "../../testdata/alltypes_plain")
dataset = lance.dataset(file_path)
print(dataset.to_table().to_pandas())
