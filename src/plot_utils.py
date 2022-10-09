import plotly.graph_objs as go
import matplotlib.pyplot as plt 

def plot_forecast(x, y_hat, y_hat_lower, y_hat_upper, y):

    x = x.apply(lambda x: x.strftime("%Y-%m-%d"))

    fig = go.Figure(
        [
            go.Scatter(
                name="y_hat",
                x=x,
                y=y_hat,
                mode="lines",
                line=dict(color="rgb(31, 119, 180)"),
            ),
            go.Scatter(
                name="Upper Bound",
                x=x,
                y=y_hat_upper,
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="Lower Bound",
                x=x,
                y=y_hat_lower,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
            ),
            go.Scatter(
                name="y",
                x=x,
                y=y,
                mode="lines",
                line=dict(color="rgb(25, 190, 70)"),
            ),
        ]
    )

    fig.update_layout(
        yaxis_title="Forecasted Sales Quantity", title="Comparison", hovermode="x"
    )

    fig.show()


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)