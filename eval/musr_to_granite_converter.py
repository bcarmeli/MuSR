# Standard
import csv
import json
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

    input_file_name = sys.argv[1]
    assert input_file_name.endswith("json"), f"Please provide a json input file. Got {input_file_name}"
    correct_only = False
    chat_format = True

    model = "microsoft/phi-4"
    dataset = "murder mysteries"
    # prompt_type = "cot+"
    # prompt_type = "cot+ s.c. 1-shot"
    prompt_type = "Phi-4-reasoning-plus"

    input_file = f'{DISTILL_FOLDER}/{input_file_name}'
    # Output file is jsonl format
    output_file = f'{GRANITE_LCOT_FOLDER}/{input_file_name}l'

    input_data = read_input_to_dict(input_file)
    output_data = []


    examples = input_data[model][dataset][prompt_type]["examples"]
    for d_ins in examples:
        assert len(d_ins) == 1, f'Several answers is not yet supported'
        d_in = d_ins[0]
        if isinstance(list(), type(d_in)): d_in = d_in[0]
        if chat_format:
            content = {"conversations":
                [
                    {"role": "user", "content": d_in["prompt"]},
                    {
                        "role": "assistant",
                        "content": d_in["output"],
                    },
                ]
            }
        else:
            content = {
                "problem": d_in["prompt"],
                "response": d_in["output"]
            }

        d_out = {
            "problem_id": d_in["qidx"],  # str(uuid.uuid4()),
            "source": f'{model}_{dataset}_{prompt_type}',
            "solution": "",
            **content,
            "ground_truth": d_in["gold_answer"],
            "correct": d_in["correct"],
        }
        if correct_only and not bool(d_in["correct"]): continue
        output_data.append(d_out)

    jsonfile = jsonlines.open(output_file, "w")
    for d in output_data:
        jsonfile.write(d)
    # jsonfile.write(psg_url_mapping)
    print(f"Wrote {len(output_data)} records to {output_file}")


if __name__ == "__main__":
    main()
