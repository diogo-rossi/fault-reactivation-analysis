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
        "average": [0.30, 1.00, 0.80, 2.00, 30.00, 25.00],
        "stddevi": [0.05, 0.10, 0.10, 0.80, 3.00, 5.00],
        "minimum": [0.05, 0.05, 0.60, 0.00, 0.00, 5.00],
        "maximum": [0.70, 1.70, 1.00, 5.00, 50.00, 45.00],
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
        "depth": [100, 200, 1850, 3000, 3500],
        "distr": ["Normal", "Normal", "Normal", "Normal", "Normal"],
        "average": [24.50, 25.00, 25.50, 25.60, 25.70],
        "stddevi": [1.715, 1.750, 1.785, 1.792, 1.799],
        "minimum": [20, 20, 20, 20, 20],
        "maximum": [27, 27, 27, 27, 27],
    },
)
