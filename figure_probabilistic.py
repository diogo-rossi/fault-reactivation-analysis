from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from plotly.graph_objects import Contour, Figure, Histogram, Scatter
from plotly.graph_objs.scatter import Line
from plotly.subplots import make_subplots

axeskwargs = {
    "mirror": "allticks",
    "ticks": "inside",
    "showgrid": True,
    "showline": True,
    "showticklabels": True,
    "title_standoff": 2,
}


@dataclass
class ProbabilisticAnalisisFigure:
    num_realizations: int
    num_bins: int

    def __post_init__(self) -> None:

        self.fig: Figure = make_subplots(
            rows=2,
            cols=4,
            horizontal_spacing=0.05,
            vertical_spacing=0.075,
        )

        self.rows = [1, 1, 1, 2, 2, 2, 1]
        self.cols = [1, 2, 3, 1, 2, 3, 4]

    def add_SF_hist(self, hist: NDArray[float64], fp: float):
        name = "Security Factor - SF"
        row, col = 1, 4
        max_value = float(int(min([hist.max(), 10])))
        data_low = hist[hist <= 1.0]
        data_high = hist[hist > 1.0]
        xbins = dict(start=0, end=max_value, size=max_value / self.num_bins)
        self.fig.update_yaxes(
            row=row,
            col=col,
            title_text=f"Frequency",
            **axeskwargs,
        )
        self.fig.update_xaxes(
            row=row,
            col=col,
            title_text=name,
            range=[0.0, max_value],
            **axeskwargs,
        )
        self.fig.add_trace(
            Histogram(x=data_low, name="SF < 1", xbins=xbins, marker_color="salmon"),
            row=row,
            col=col,
        )
        self.fig.add_trace(
            Histogram(name="SF > 1", x=data_high, xbins=xbins),
            row=row,
            col=col,
        )
        self.fig.add_vline(
            row=row,  # type: ignore
            col=col,  # type: ignore
            x=1.0,
            line_width=3,
            line_dash="dash",
            line_color="red",
            opacity=1,
        )
        self.add_annotation(1, 4, hist, extra=f"\nPf = {fp:.2f} %")

    def add_var_hist(self, i, data, hist):
        self.fig.update_yaxes(
            row=self.rows[i],
            col=self.cols[i],
            title_text=f"Frequency",
            **axeskwargs,
        )
        self.fig.update_xaxes(
            row=self.rows[i],
            col=self.cols[i],
            title_text=data[0],
            **axeskwargs,
        )
        self.fig.add_trace(
            Histogram(
                x=hist,
                nbinsx=self.num_bins,
                name=data[0],
            ),
            row=self.rows[i],
            col=self.cols[i],
        )
        self.add_annotation(self.rows[i], self.cols[i], hist)

    def add_annotation(self, row, col, hist, extra=""):
        mini = np.min(hist)
        maxi = np.max(hist)
        mean = np.mean(hist)
        stdev = np.std(hist)
        counts, bin_edges = np.histogram(hist, bins=self.num_bins)
        max_bin_index = np.argmax(counts)
        mode = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        self.fig.add_annotation(
            text=f"Mean = {mean:.2f}<br>"
            f"Std dev = {stdev:.2f}<br>"
            f"Minimum = {mini:.2f}<br>"
            f"Maximum = {maxi:.2f}<br>"
            f"Mode = {mode:.2f}<br>"
            f"Samples = {self.num_realizations}<br>"
            f"{extra}",
            xref="x2 domain",
            yref="y2 domain",
            x=0.98,
            y=0.98,
            showarrow=False,
            align="right",
            bordercolor="black",
            borderwidth=1,
            bgcolor="white",
            row=row,
            col=col,
        )

    def update_layout(self):
        self.fig.update_layout(
            template="simple_white",
            plot_bgcolor="white",
            barmode="overlay",
            height=650,
            margin=dict(l=50, r=40, t=30, b=0),
            showlegend=True,
            dragmode="zoom",
        )
        self.fig.update_traces(marker_line_width=0, selector=dict(type="histogram"))

    def update_contour_axes(self, dPmin, dPmax, zmin, zmax):
        self.fig.update_xaxes(
            row=2,
            col=4,
            title_text=f"Pressure variation - ΔP [MPa]",
            range=[dPmin, dPmax],
            **axeskwargs,
        )
        self.fig.update_yaxes(
            row=2,
            col=4,
            title_text=f"Injection layer depth [m]",
            range=[zmin, zmax],
            **axeskwargs,
        )

    def add_fp_contours(self, x, y, z):
        contour = Contour(
            x=x,
            y=y,
            z=z,
            colorscale="Turbo",
            ncontours=20,
            contours=dict(
                coloring="fill",
                showlines=False,
            ),
            colorbar=dict(
                x=1.01,
                y=0.23,
                len=0.5,
                title=dict(
                    text="Probability of fault reactivation (failure) - Pf (%)",
                    side="right",
                ),
            ),
            name="SF map",
            showlegend=False,
        )
        X, Y = np.meshgrid(x, y[0 : -1 : int(len(y) / 50)])
        mesh = Scatter(
            x=X.flatten(),
            y=Y.flatten(),
            mode="markers",
            marker=dict(size=5, color="rgba(0,0,0,0)"),
            visible=True,
            showlegend=False,
            name="ΔP vs z",
        )
        self.fig.add_trace(contour, row=2, col=4)
        self.fig.add_trace(mesh, row=2, col=4)

    def update_current_point(self, p, z):
        self.fig.add_hline(z, 2, 4, line_color="black", line_width=2)  # type: ignore
        self.fig.add_vline(p, 2, 4, line_color="black", line_width=2)  # type: ignore
        self.fig.add_trace(
            row=2,
            col=4,
            trace=Scatter(
                x=[p],
                y=[z],
                name="Current",
                line=Line(color="black", width=1),
                showlegend=False,
            ),
        )
