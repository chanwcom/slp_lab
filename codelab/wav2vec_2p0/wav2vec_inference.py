# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
from transformers import pipeline

# Custom imports
import sample_util_solution
sample_util = sample_util_solution

db_top_dir = "/mnt/data/database/libri_speech_webdataset_new_oct_2025"

test_clean_top_dir = os.path.join(db_top_dir, "test-clean")
test_other_top_dir = os.path.join(db_top_dir, "test-other")


# TODO Complete the following parts:
test_clean_dataset = sample_util.make_dataset( ,  )
test_other_dataset = sample_util.make_dataset( ,  )

transcriber = pipeline(
    ,
    model=""
)
# End of TODO

# Function to write REF/HYP pairs to a file
def write_results(dataset, transcriber, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for data in dataset:
            ref = data["labels"]
            # TODO complete the following part
            hyp = transcriber()[""]
            # End of TODO
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")  # double newline for readability

# Write test_clean_dataset
write_results(test_clean_dataset, transcriber, "test_clean_result.txt")

# Write test_other_dataset
write_results(test_other_dataset, transcriber, "test_other_result.txt")
