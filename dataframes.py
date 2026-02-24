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
        "distr": ["Normal", "Normal", "Normal", "Normal", "Normal", "Normal"],
        "average": [0.25, 1.00, 0.50, 2, 30.0, 25.0],
        "stddevi": [0.05, 0.10, 0.10, 0.1, 5, 5],
        "minimum": [0.05, 0.05, 0.10, 1, 5, 10],
        "maximum": [0.70, 1.70, 1.00, 5, 50, 45],
    },
)

layers_df = DataFrame(
    {
        "layer": [
            "First layer name",
            "Second layer name",
            "Third layer name",
            "Fourth layer name",
            "Fifth layer name",
        ],
        "depth": [100, 200, 800, 1800, 2000],
        "distr": ["Normal", "Normal", "Normal", "Normal", "Normal"],
        "average": [24, 25, 23, 25, 25],
        "stddevi": [2, 2, 2, 2, 2],
        "minimum": [20, 20, 20, 20, 20],
        "maximum": [27, 27, 27, 27, 27],
    },
)
