import matplotlib.pyplot as plt

log_file_name = "~/log/dlinknet_new_lr_decoder_total_train.log"
loss_list = []
with open(log_file_name, "r") as f:
    logs_lines = f.readlines()
    for line in logs_lines:
        if "train_loss" in line:
            temp = line.strip().split(":")
            loss_list.append(float(temp[-1]))
f.close()
x = [x for x in range(len(loss_list))]
plt.plot(x, loss_list, "r")
plt.show()