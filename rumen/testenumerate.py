with open('test.txt', 'r') as f:
    for idx, line in enumerate(f):
        print(f'Line number {idx}: {line.strip()}')
