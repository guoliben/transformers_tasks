def generate_samples():
    sample1 = {'id': 1, 'name': 'Sample 1'}
    yield sample1
    sample2 = {'id': 2, 'name': 'Sample 2'}
    yield sample2
    sample3 = {'id': 3, 'name': 'Sample 3'}
    yield sample3

# 使用生成器函数
samples = generate_samples()

print(samples)
for sample in samples:
    print(sample)
