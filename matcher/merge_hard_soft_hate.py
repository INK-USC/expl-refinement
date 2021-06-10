import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--soft_path',
                        default=None, type=str, required=True, help='Advice file path')
parser.add_argument('--hard_path',
                        default=None, type=str, required=True, help='Advice file path')
parser.add_argument('--nonhate_path',
                        default=None, type=str, required=True, help='Advice file path')
parser.add_argument('--output_path',
                        default=None, type=str, required=True, help='Advice file path')
args = parser.parse_args()

soft_file = args.soft_path
hard_file = args.hard_path
nonhate_file = args.nonhate_path
merge_file = args.output_path

with open(hard_file, 'r', encoding='utf-8') as f:
    hard_lines = f.readlines()

with open(soft_file, 'r', encoding='utf-8') as f:
    soft_lines = f.readlines()

with open(nonhate_file, 'r', encoding='utf-8') as f:
    nonhate_lines = f.readlines()

def merge_two(merge_file):
    merge_lines = []

    # replace confidence column
    # hard and soft: conf = 1, pure soft: conf = 0
    hard_and_soft = 0
    pure_soft = 0
    for soft_line in soft_lines:
        sentence, advice, instance_label = soft_line.strip('\n').split('\t')
        if soft_line in hard_lines:
            line = '\t'.join([sentence, advice, instance_label, '1'])
            hard_and_soft += 1
        else:
            line = '\t'.join([sentence, advice, instance_label, '0'])
            pure_soft += 1
        merge_lines.append(line)
    
    return merge_lines

merge_lines = merge_two(merge_file)
for nonhate_line in nonhate_lines:
    if nonhate_line[-1] == '\n':
        nonhate_line = nonhate_line[:-1]
    line = '\t'.join([nonhate_line, '1'])
    merge_lines.append(line)

with open(merge_file, 'w', encoding='utf-8') as f:
    for line in merge_lines:
        f.write(line + '\n')