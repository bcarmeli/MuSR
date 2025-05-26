# Standard
import csv
import json
import re
import sys
import uuid

# Third Party
import jsonlines

from src.utils.paths import DISTILL_FOLDER, GRANITE_LCOT_FOLDER


def read_input_to_dict(input_file, skip_header=False, fieldnames=()):
    if input_file.endswith('jsonl'):
        with jsonlines.open(input_file, 'r') as f:
            input_data = [line for line in f]
    elif input_file.endswith('json'):
        with open(input_file, 'r') as f:
            input_data = json.load(f)
    elif input_file.endswith('csv'):
        csvfile = open(input_file, 'r')
        reader = csv.DictReader(csvfile, fieldnames)
        if skip_header:
            next(reader)
        input_data = [line for line in reader]
    elif input_file.endswith('txt'):
        with open(input_file, 'r') as f:
            input_data = f.readlines()
    else:
        assert False, f'Unknown input file format {input_file}. File needs to end with either csv, json or jsonl.'
    print(f'Read {len(input_data)} samples from {input_file}')
    return input_data

def main():

    input_file = sys.argv[1]
    file_extension = input_file.split(".")[-1]
    output_file = input_file.replace(f".{file_extension}", f"_results.{file_extension}")

    input_data = read_input_to_dict(input_file)

    # assert len(input_data.keys()) == 1, f"File should contain just one model"
    # model = list(input_data.keys())[0]
    # assert len(input_data[model].keys()) == 1, f"File should contain just one dataset"
    results = []
    models = list(input_data.keys())
    for model in models:
        dataset = list(input_data[model].keys())[0]
        prompt_types = list(input_data[model][dataset].keys())
        for prompt_type in prompt_types:
            examples = input_data[model][dataset][prompt_type]["examples"]
            correct = incorrect = 0
            for d_ins in examples:
                assert len(d_ins) == 1, f'Several answers is not yet supported'
                d_in = d_ins[0]
                if isinstance(list(), type(d_in)): d_in = d_in[0]
                # system_part, user_part, assistant_part = parse__output(d_in["output"])
                if bool(d_in["correct"]):
                    correct += 1
                else:
                    incorrect += 1
            results.append({"model": model, "dataset": dataset, "prompt": prompt_type, "correct": correct, "incorrect": incorrect, "accuracy": correct/(correct + incorrect)})

    jsonfile = jsonlines.open(output_file, "w")
    jsonfile.write(results)
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
