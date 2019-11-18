import os
import matplotlib.pyplot as plt

# data directory
data_dir = os.path.join("data", "gtsrb-german-traffic-sign", "Train")
# print folders in data directory
print(os.listdir(data_dir))

# signs
signs = os.listdir(data_dir)

# number of samples
num_samples = list(map(lambda s: len(os.listdir(os.path.join(data_dir, s))), signs))

print('total number of samples:', sum(num_samples))

# plot
plt.figure(figsize=(8, 2), dpi=160)
plt.bar(range(1, len(signs) + 1), num_samples, color='xkcd:warm purple')
plt.xticks(range(1, len(signs) + 1), signs, rotation=90)
plt.xlabel('sign')
plt.ylabel('number of samples')
plt.title('distribution of samples')
plt.show()
