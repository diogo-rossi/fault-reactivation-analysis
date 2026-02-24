from __future__ import annotations

from typing import Any, Literal, overload

from pandas import DataFrame, Series


class InjectionLayerTable(DataFrame):
    parameter: Series[str]
    distr: Series[str]
    average: Series[float]
    stddevi: Series[float]
    minimum: Series[float]
    maximum: Series[float]

    @overload
    def __getitem__(self, n: Literal["parameter"]) -> Series[str]: ...

    @overload
    def __getitem__(self, n: Literal["distr"]) -> Series[str]: ...

    @overload
    def __getitem__(self, n: Literal["average"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["stddevi"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["minimum"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["maximum"]) -> Series[float]: ...

    def __getitem__(self, n) -> Any:
        return super().__getitem__(n)

    def get_value_from_df(
        self,
        line_name: Literal[
            "Ko", "Ka", "biot", "cohesion", "friction", "inclination", "overpressure"
        ],
        column_name: Literal["average", "stddevi", "minimum", "maximum"],
    ):
        return self[column_name][
            [line_name.lower() in line.lower() for line in self.iloc[:, 0]]
        ].to_list()[0]

    def get_mean_value_from_df(
        self,
        line_name: Literal[
            "Ko", "Ka", "biot", "cohesion", "friction", "inclination", "overpressure"
        ],
    ):
        return self.get_value_from_df(line_name, "average")


class LayerTable(DataFrame):
    layer: Series[str]
    depth: Series[float]
    distr: Series[str]
    average: Series[float]
    stddevi: Series[float]
    minimum: Series[float]
    maximum: Series[float]

    @overload
    def __getitem__(self, n: Literal["layer"]) -> Series[str]: ...

    @overload
    def __getitem__(self, n: Literal["depth"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["distr"]) -> Series[str]: ...

    @overload
    def __getitem__(self, n: Literal["average"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["stddevi"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["minimum"]) -> Series[float]: ...

    @overload
    def __getitem__(self, n: Literal["maximum"]) -> Series[float]: ...

    def __getitem__(self, n) -> Any:
        return super().__getitem__(n)
