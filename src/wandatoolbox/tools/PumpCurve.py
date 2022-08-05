import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import List


class PumpCurve:
    def __init__(
        self,
        filepath: str,
        export_columns: List = ['Discharge', 'Pump head', 'Efficiency'],
        qh_name: str = "QH.csv",
        qh_dict: dict = None,
        qe_name: str = "QE.csv",
        qe_dict: dict = None,
        delimiter: str = ";",
        decimal: str = ".",
        q0: list = None,
        Qkey: str = "Q",
        Hkey: str = "H",
        Ekey: str = "E",
    ):
        """Class to work with pump curves in CSV format.

        <Additional comments>.

        :param str filepath: Directory to retrieve the documents from.
        :param List[str] export_columns: Output format name.

        """

        self.filepath = filepath
        # Define csv names
        self.qh_name = os.path.join(self.filepath, qh_name)
        self.qe_name = os.path.join(self.filepath, qe_name)
        # Define transformation dictionaries
        self.qe_dict = qe_dict
        self.qh_dict = qh_dict
        # Keys
        self.Qkey = Qkey
        self.Hkey = Hkey
        self.Ekey = Ekey
        self.export_columns = export_columns
        # Define curves
        self.df_qh = pd.read_csv(self.qh_name, delimiter=delimiter, decimal=decimal)
        self.df_qe = pd.read_csv(self.qe_name, delimiter=delimiter, decimal=decimal)
        # Set transformations.
        self.df_qh = self.variable_transformation(
            df=self.df_qh, transform_dict=self.qh_dict
        )
        self.df_qe = self.variable_transformation(
            df=self.df_qe, transform_dict=self.qe_dict
        )
        # Interpolate to qint if present
        self.q0 = q0
        self.qint = None
        self.hint = None
        self.eint = None

    def variable_transformation(self, df: pd.DataFrame, transform_dict: dict = None):
        """

        Parameters
        ----------
        df: pd.Dataframe to convert
        transform_dict: dict with keys of the dataframe to convert.

        Returns
        -------
        df: pd.Dataframe converted

        """
        if transform_dict:
            for transform_key in list(transform_dict.keys()):
                if df.get(transform_key) is not None:
                    df[transform_key] = df[transform_key] * transform_dict[transform_key]
            return df
        else:
            return df

    def interpolate_curves(self):
        if self.q0 is None:
            Nelements = np.max([len(self.df_qh[self.Qkey]), len(self.df_qe[self.Qkey])])
            Qmin = np.r_[
                self.df_qh[self.Qkey].values, self.df_qe[self.Qkey].values
            ].min()
            if Qmin <= 0:
                Qmin = 1e-6

            Qmax = np.r_[
                self.df_qh[self.Qkey].values, self.df_qe[self.Qkey].values
            ].max()
            self.qint = np.linspace(Qmin, Qmax, Nelements)
        else:
            self.qint = self.q0
        # Interpolate H
        self.hint = np.interp(self.qint, self.df_qh[self.Qkey], self.df_qh[self.Hkey])
        # Interpolate E
        self.eint = np.interp(self.qint, self.df_qe[self.Qkey], self.df_qe[self.Ekey])

    def export_csv(self):
        # Interpolate curves
        self.interpolate_curves()
        # Construct dataframe
        df = pd.DataFrame(
            np.c_[self.qint, self.hint, self.eint], columns=self.export_columns
        )
        # Export
        df.to_csv(
            os.path.join(self.filepath, "export.csv"), decimal=",", sep=";", index=False
        )

    def plot(self, figname: str = "export_fig.png"):
        plt.figure(figsize=(10 / 2.54, 8 / 2.54))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(self.df_qh[self.Qkey], self.df_qh[self.Hkey], "o", color="k")
        ax2.plot(self.df_qe[self.Qkey], self.df_qe[self.Ekey], "s", color="b")
        ax1.plot(self.qint, self.hint, "-k")
        ax2.plot(self.qint, self.eint, "--b")
        ax1.set_xlabel(r"$Q \ (m^3/s)$")
        ax1.set_ylabel(r"$H \ (m)$")
        ax2.set_ylabel(r"$E \ (\%)$")
        ax2.yaxis.label.set_color("b")
        ax2.tick_params(axis="y", colors="b")
        plt.tight_layout()
        plt.savefig(os.path.join(self.filepath, figname))


if __name__ == "__main__":
    # Path to the QH and QE csv
    filepath = r""
    pump_curve = PumpCurve(
        filepath=filepath,
        delimiter=r'\t',
        decimal='.'
    )
    pump_curve.export_csv()
    pump_curve.plot()
