# Standard
import csv
import json
import re
import sys
import uuid

# Third Party
import jsonlines
from transformers import AutoTokenizer


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

def count_words_in_text(text):
    words = re.findall(r"\b\w+\b", text)
    return len(words)

def count_words(text: str, think_counts: list, answer_counts: list):
    think_text, answer_text = text.split("</think>")
    think_words = count_words_in_text(think_text)
    answer_words = count_words_in_text(answer_text)

    think_counts.append(think_words)
    answer_counts.append(answer_words)
    return think_counts, answer_counts


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    assert input_file.endswith("jsonl"), f"Please provide a jsonl input file. Got {input_file}"
    # Output file is jsonl format

    input_data = read_input_to_dict(input_file)
    output_data = []
    think_counts = []
    answer_counts = []
    missing_think_counter = 0

    for d_in in input_data:
        assistant_text = d_in["messages"][1]['content']
        assistant_text = assistant_text.replace("<>", "")
        if not ("<think>" in assistant_text and "</think>" in assistant_text):
            missing_think_counter += 1
            continue
        think_counts, answer_counts = count_words(assistant_text, think_counts, answer_counts)        # Remove the thik part from the assistant role
        assistant_text = assistant_text.replace("<think>", "<think>\n")
        assert "\n</think>" not in assistant_text
        assistant_text= assistant_text.replace("</think>", "\n</think>\n\n")
        # d_in["messages"][1]['content'] = d_in["messages"][1]['content'].split("</think>")[-1]
        d_in["messages"][1]['content'] = assistant_text
        d_out = {
            "id": d_in["id"] if "id" in d_in.keys() else str(uuid.uuid4()),
            "source": d_in["source"],
            "messages": d_in["messages"],
        }
        output_data.append(d_out)
    jsonfile = jsonlines.open(output_file, "w")
    for d in output_data:
        jsonfile.write(d)
    print(f"Wrote {len(output_data)} records to {output_file}")
    print(f"Found {missing_think_counter} samples without '<think>'")
    print(f"Average think word: {sum(think_counts)/len(think_counts)}")
    print(f"Average answer word: {sum(answer_counts) / len(answer_counts)}")



if __name__ == "__main__":
    main()
