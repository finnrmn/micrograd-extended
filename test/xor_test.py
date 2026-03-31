from micograd import MLP, Layer, Activation

# Dataset: XOR problem
# inputs
X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
# labels: 1.0 = true, -1.0 = false
y = [1.0, -1.0, -1.0, 1.0]

# MLP: 2 inputs -> hidden layer 4 -> hidden layer 4 -> 1 output
model = MLP(
    Layer(2, 4, Activation.Tanh),
    Layer(4, 4, Activation.Tanh),
    Layer(4, 1, Activation.Linear)
)
print("")
print(model)
print(f"Parameter: {len(model.parameters())}")

# Training Loop
learning_rate = 0.05
epochs = 100

print(f"\nTraining Loop:")
for epoch in range(epochs):
    # forward pass
    y_pred = [model(x) for x in X]

    # MSE loss
    loss = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y))

    # backward pass
    model.zero_grad()
    loss.backward()

    # gradient descent update
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"   Epoch {epoch:3d} | Loss: {loss.data:.8f}")

# Final predictions
print("\nFinal predictions:")
for x, yt in zip(X, y):
    yp = model(x)
    print(f"   input {x} -> predicted {yp.data:+.8f} | target {yt:+.1f}")

print("")
# Graph Graph (optional)

# from utils.viz import draw_dot
# draw_dot(yp)
