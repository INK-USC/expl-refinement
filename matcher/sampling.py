import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default=None, type=str, required=True, help='Advice file path')
    parser.add_argument('--number', 
                        default=None, type=int, required=True, help='Advice file path')
    args = parser.parse_args()

    with open(args.path, encoding='utf-8') as f:
        advice_lines = f.readlines()

    print(len(advice_lines))
    random.shuffle(advice_lines)
    lines = advice_lines[:args.number]
    print(len(lines))

    with open(args.path + '_' + str(args.number), 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)
