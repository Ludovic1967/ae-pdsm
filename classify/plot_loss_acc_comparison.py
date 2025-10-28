import re
import matplotlib.pyplot as plt


def parse_log(log_path):
    losses = []
    accs = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取 loss
            loss_match = re.search(r'Class Loss: ([\d.]+)', line)
            if loss_match:
                losses.append(float(loss_match.group(1)))
            # 提取 Top-1 acc
            acc_match = re.search(r'@Top-1 Score: ([\d.]+)', line)
            if acc_match:
                accs.append(float(acc_match.group(1)))
    return losses, accs


# 路径修改为你训练过程中保存的日志
log_cbam = './logs/resnet18_cbam.log'
log_base = './logs/resnet34_baseline.log'

loss_cbam, acc_cbam = parse_log(log_cbam)
loss_base, acc_base = parse_log(log_base)

print(len(loss_base), len(loss_cbam))
print(len(acc_base), len(acc_cbam))

# 你也可以选择每 N 个 batch 平滑一次 loss
def smooth(data, window=2):
    return [sum(data[i:i+window])/window for i in range(0, len(data)-window+1)]

# 绘图
plt.figure(figsize=(12, 5))

# Loss 对比
plt.subplot(1, 2, 1)
# plt.yscale('log')
plt.plot(smooth(loss_cbam), label='CBAM')
plt.plot(smooth(loss_base), label='Baseline')
plt.title('Training Loss (Smoothed)')
plt.xlabel('Batch (smoothed)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy 对比
plt.subplot(1, 2, 2)
# plt.yscale('log')
plt.plot(acc_cbam, label='CBAM')
plt.plot(acc_base, label='Baseline')
plt.title('Top-1 Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
