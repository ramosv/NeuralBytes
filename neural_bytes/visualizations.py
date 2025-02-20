import matplotlib.pyplot as plt

def visualize_nn(cost_history, nn, final_params, X, Y):
    plt.figure(figsize=(6, 4))
    plt.plot(cost_history, color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title(f"Loss Curve over {nn.epochs} epochs")
    plt.savefig(f"plots/loss_curve_{nn.epochs}_epochs_{nn.lr}_lr.png")
    plt.show()

    # adding extra space so its not soo squeezed
    x_min = min(X[0]) - 0.5
    x_max = max(X[0]) + 0.5
    y_min = min(X[1]) - 0.5
    y_max = max(X[1]) + 0.5

    num_points = 100
    # evenly spaced grid for x and y
    grid_x = []
    for i in range(num_points):
        grid_x.append(x_min + i * (x_max - x_min) / (num_points - 1))

    grid_y = []
    for j in range(num_points):
        grid_y.append(y_min + j * (y_max - y_min) / (num_points - 1))

    xx = []
    yy = []
    for j in range(num_points):
        row_x = []
        row_y = []
        for i in range(num_points):
            row_x.append(grid_x[i])
            row_y.append(grid_y[j])
        xx.append(row_x)
        yy.append(row_y)

    # predictions for each point
    Z = []
    for j in range(num_points):
        row_outputs = []
        for i in range(num_points):
            x_val = grid_x[i]
            y_val = grid_y[j]
            input_point = [[x_val], [y_val]]
            output = nn.predict(input_point, final_params)
            row_outputs.append(output[0][0])
        Z.append(row_outputs)

    #  print min/max predictions
    flat_values = []
    for row in Z:
        for val in row:
            flat_values.append(val)
    print(f"Min pred: {min(flat_values)} | Max pred: {max(flat_values)}")

    # decision boundary
    plt.figure(figsize=(6, 5))

    contour_levels = []
    for k in range(100):
        contour_levels.append(k / (100 - 1))
    plt.contourf(xx, yy, Z, levels=contour_levels, cmap='coolwarm', alpha=0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)

    # the raining points
    for idx in range(len(X[0])):
        x_pt = X[0][idx]
        y_pt = X[1][idx]
        label = Y[0][idx]
        color = 'red' if label == 1 else 'blue'
        plt.scatter(x_pt, y_pt, color=color, edgecolor='black', s=100, zorder=2)

    plt.title(f"Decision Boundary with: {nn.lr} learning rate")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(label="Confidence on prediction")
    plt.savefig(f"plots/decision_boundary_{nn.epochs}_epochs_{nn.lr}_lr.png")
    plt.show()