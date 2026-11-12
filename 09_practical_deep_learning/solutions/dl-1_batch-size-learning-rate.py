batch_sizes, learning_rates = [128, 2048], [1e-3, 2e-3, 4e-3, 8e-3]
results = {}
for batch_size in batch_sizes:
    for lr in learning_rates:
        _, history = fit(batch_size, lr, epochs=4)
        results[batch_size, lr] = history[-1]
        print(f"B={batch_size:4d}, lr={lr:.0e}: "
              f"accuracy {history[-1, 1]:.1%} in {history[-1, 0]:.0f}s")

plt.figure(figsize=(5, 3.5))
for batch_size in batch_sizes:
    accuracies = [results[batch_size, lr][1] for lr in learning_rates]
    plt.plot(learning_rates, accuracies, "o-", label=f"B = {batch_size}")
plt.xscale("log")
plt.xlabel("Learning rate")
plt.ylabel("Validation accuracy")
plt.legend()

# 1. The best learning rate moves much less than the x4 of the sqrt(B) rule, and
#    it changes from one run to the next. The scaling rules give a starting point:
#    the learning rate must still be tuned.
# 2. B = 2048 is barely faster per epoch: the GPU was already busy at B = 128
#    (Section 2.3). And it is much worse after 4 epochs: it makes 16 times fewer
#    steps, far beyond the critical batch size. Use the smallest batch size that
#    keeps the GPU busy.
