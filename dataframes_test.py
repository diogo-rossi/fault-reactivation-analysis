from pandas import DataFrame

injection_layer_df = DataFrame(
    {
        "parameter": [
            "Initial lateral stress multiplier - Ko [-]",
            "Active lateral stress multiplier - Ka [-]",
            "Biot coefficient - α [-]",
            "Fault cohesion - c [MPa]",
            "Fault friction angle - ϕ [°]",
            "Fault inclination angle - θ [°]",
        ],
        "distr": ["Normal", "Normal", "Triangular", "Normal", "Normal", "Triangular"],
        "average": [0.28205128205128205, 0.28205128205128205, 0.80, 4.402, 31.5, 25.0],
        "stddevi": [0.12359550561797752, 0.12359550561797752, 0.10, 1.094, 3.15, 1.00],
        "minimum": [0.0, 0.0, 0.60, 00, 00, 5],
        "maximum": [1.0, 1.0, 1.00, 10, 50, 45],
    },
)

layers_df = DataFrame(
    {
        "layer": [
            "First layer name",
            "Second layer name",
            "Third layer name",
            "Fourth layer name",
        ],
        "depth": [2000, 3000, 5000, 5500],
        "distr": ["Normal", "Normal", "Normal", "Normal"],
        "average": [11.2, 25.0, 21.0, 25.0],
        "stddevi": [0.10, 1.75, 1.47, 1.75],
        "minimum": [11.1, 20.0, 20.0, 20.0],
        "maximum": [11.3, 30.0, 30.0, 30.0],
    },
)
