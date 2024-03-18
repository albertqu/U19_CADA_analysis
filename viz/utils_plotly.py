from plotly.subplots import make_subplots
import plotly.graph_objects as go


class PlotlyFig:
    # TODO: add automatic color palette

    def __init__(
        self, rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1, **kwargs
    ):
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=shared_xaxes,
            vertical_spacing=vertical_spacing,
            **kwargs,
        )
        self.fig = fig

    def plot(
        self, x, y, name="", color="blue", mode="lines+markers", row=1, col=1, **kwargs
    ):
        self.fig.add_trace(
            go.Scatter(x=x, y=y, mode=mode, name=name, line=dict(color=color)),
            row=row,
            col=col,
        )

    def show(self):
        self.fig.show()
