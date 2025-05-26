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
    input_file =  str(input_file)
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

def collect_examples(examples_node):
    examples = []

    def traverse(node):
        if isinstance(node, list):
            for child in node:
                traverse(child)
        elif isinstance(node, dict):
            examples.append(node)
        # Stop recursion if it's a leaf (non-list, non-dict)
        else:
            return

    traverse(examples_node)
    return examples

def old_parse__output(model_output):
    system_part, rest = model_output.split("guidelines:user")
    system_part = system_part.replace("systemYou", "You")
    system_part += "guidelines:"
    user_part, assistant_part = rest.split("\"assistant")
    user_part += "\""
    return system_part, user_part, assistant_part

def parse_output(model_output):
    if not "<think>" in model_output: return None
    if not "</think>" in model_output: return None

    if model_output.startswith("<think>"):
        assistant_part = model_output
    else:
        assistant_part = "<think>" + model_output.split("<think>")[-1]
        # assistant_part = "<think>" + assistant_part
    assistant_part_no_spacial_tokens =  re.sub(r'\|[^|]+\|', '', assistant_part)
    return assistant_part_no_spacial_tokens

def main():

    input_file_name = sys.argv[1]
    prompt_type = sys.argv[2]
    dataset_type = sys.argv[3]
    assert input_file_name.endswith("json"), f"Please provide a json input file. Got {input_file_name}"
    correct_only = False
    chat_format = True

    # model = "microsoft/Phi-4-reasoning-plus"
    # dataset = "murder mysteries"
    model = "microsoft/phi-4"
    dataset = "object placements"
    # prompt_type_string = "cot+ hint"
    # prompt_type = "cot+ s.c. 1-shot"
    # prompt_type = "Phi-4-reasoning-plus"
    if prompt_type == "cot_plus":
        prompt_type_string = "cot+"
        prompt_dir_name = "cot_plus"
    elif prompt_type == "cot_plus_hint":
        prompt_type_string = "cot+ hint"
        prompt_dir_name = "cot_plus_hint"
    else:
        assert False, f"Unknown prompt type {prompt_type}"

    input_file =  DISTILL_FOLDER / dataset_type / input_file_name
    # Output file is jsonl format
    output_file = GRANITE_LCOT_FOLDER/ dataset_type / prompt_dir_name / f"{input_file_name}l"

    input_data = read_input_to_dict(input_file)
    output_data = []
    skipped = 0

    examples_node = input_data[model][dataset][prompt_type_string]["examples"]
    # From some reason, the structure in the "examples" part is not consistent and qa pairs
    # may appear as standalone dicts, as dicts in list, and even as dict in list of list
    examples = collect_examples(examples_node)
    for d_in in examples:

        # assert len(d_ins) == 1, f'Several answers is not yet supported'
        # d_in = d_ins[0]
        # if isinstance(list(), type(d_in)): d_in = d_in[0]
        # system_part, user_part, assistant_part = parse__output(d_in["output"])
        # for d_in in d_ins:
        # assert len(d_in) == 1, f'Unknown format'
        # d_in = d_in[0]
        user_part = d_in["prompt"]
        # assistant_part = parse_output(d_in["output"])
        assistant_part = d_in["output"]
        if assistant_part == None:
            skipped += 1
            continue
        if chat_format:
            content = {"messages":
                [
                    # {"role": "system", "content": system_part},
                    {"role": "user", "content": user_part},
                    {"role": "assistant", "content": assistant_part},
                ]
            }
        else:
            content = {
                "problem": user_part,
                "response": assistant_part
            }

        d_out = {
            "id": str(uuid.uuid4()),
            "qidx": d_in["qidx"],
            "qhash": d_in["qhash"],
            "source": f'{model}_{dataset}_{prompt_type_string}',
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
    print(f"Skip {skipped} records due to bad assistant answer format")



if __name__ == "__main__":
    main()
