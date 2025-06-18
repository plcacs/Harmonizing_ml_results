# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import string
import hashlib
import warnings
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nevergrad.common.typing as tp
from . import utils
from .exporttable import export_table

# pylint: disable=too-many-locals

_DPI: int = 250
no_limit: bool = False
pure_algorithms: tp.List[str] = []

# %% Basic tools

def compactize(name: str) -> str:
    if len(name) < 70:
        return name
    hashcode: str = hashlib.md5(bytes(name, "utf8")).hexdigest()
    name = re.sub(r"\([^()]*\)", "", name)
    mid: int = 35
    name = name[:mid] + hashcode + name[-mid:]
    return name

def _make_style_generator() -> tp.Iterator[str]:
    lines = itertools.cycle(["-", "--", ":", "-."])  # 4
    markers = itertools.cycle("ov^<>8sp*hHDd")  # 13
    colors = itertools.cycle("bgrcmyk")  # 7
    return (l + m + c for l, m, c in zip(lines, markers, colors))

class NameStyle(tp.Dict[str, tp.Any]):
    """Provides a style for each name, and keeps to it"""

    def __init__(self) -> None:
        super().__init__()
        self._gen: tp.Iterator[str] = _make_style_generator()

    def __getitem__(self, name: str) -> tp.Any:
        if name not in self:
            super().__setitem__(name, next(self._gen))
        return super().__getitem__(name)

def _make_winners_df(df: pd.DataFrame, all_optimizers: tp.List[str]) -> utils.Selector:
    """Finds mean loss over all runs for each of the optimizers, and creates a matrix
    winner_ij = 1 if opt_i is better (lower loss) then opt_j (and .5 for ties)
    """
    if not isinstance(df, utils.Selector):
        df = utils.Selector(df)
    all_optim_set: tp.Set[str] = set(all_optimizers)
    assert all(x in all_optim_set for x in df.unique("optimizer_name"))
    assert all(x in df.columns for x in ["optimizer_name", "loss"])
    winners: utils.Selector = utils.Selector(index=all_optimizers, columns=all_optimizers, data=0.0)
    grouped = df.loc[:, ["optimizer_name", "loss"]].groupby(["optimizer_name"]).mean()
    df_optimizers: tp.List[str] = list(grouped.index)
    values: np.ndarray = np.array(grouped)
    diffs: np.ndarray = values - values.T
    # loss_ij = 1 means opt_i beats opt_j once (beating means getting a lower loss/regret)
    winners.loc[df_optimizers, df_optimizers] = (diffs < 0) + 0.5 * (diffs == 0)
    return winners

def aggregate_winners(
    df: utils.Selector, categories: tp.List[str], all_optimizers: tp.List[str]
) -> tp.Tuple[utils.Selector, int]:
    """Computes the sum of winning rates on all cases corresponding to the categories

    Returns
    -------
    Selector
        the aggregate
    int
        the total number of cases
    """
    if not categories:
        return _make_winners_df(df, all_optimizers), 1
    subcases: tp.List[tp.Any] = df.unique(categories[0])
    if len(subcases) == 1:
        return aggregate_winners(df, categories[1:], all_optimizers)
    iterdf, iternum = zip(
        *(
            aggregate_winners(
                df.loc[
                    df.loc[:, categories[0]]
                    == val
                    # if categories[0] != "budget"
                    # else df.loc[:, categories[0]] <= val
                ],
                categories[1:],
                all_optimizers,
            )
            for val in subcases
        )
    )
    return sum(iterdf), sum(iternum)  # type: ignore

def _make_sorted_winrates_df(victories: pd.DataFrame) -> pd.DataFrame:
    """Converts a dataframe counting number of victories into a sorted
    winrate dataframe. The algorithm which performs better than all other
    algorithms comes first. When you do not play in a category, you are
    considered as having lost all comparisons in that category.
    """
    assert all(x == y for x, y in zip(victories.index, victories.columns))
    winrates: pd.DataFrame = victories / (victories + victories.T).max(axis=1)
    # mean_win = winrates.quantile(.05, axis=1).sort_values(ascending=False)
    mean_win: pd.Series = winrates.mean(axis=1).sort_values(ascending=False)
    return winrates.loc[mean_win.index, mean_win.index]

# %% plotting functions

def remove_errors(df: pd.DataFrame) -> utils.Selector:
    df = utils.Selector(df)
    if "error" not in df.columns:  # backward compatibility
        return df  # type: ignore
    # errors with no recommendation
    nandf: utils.Selector = df.select(loss=np.isnan)
    for row in nandf.itertuples():
        msg: str = f'Removing "{row.optimizer_name}"'
        msg += f" with dimension {row.dimension}" if hasattr(row, "dimension") else ""
        msg += f': got error "{row.error}"' if isinstance(row.error, str) else "recommended a nan"
        warnings.warn(msg)
    # error with recorded recommendation
    handlederrordf: utils.Selector = df.select(error=lambda x: isinstance(x, str) and x, loss=lambda x: not np.isnan(x))
    for row in handlederrordf.itertuples():
        warnings.warn(
            f'Keeping non-optimal recommendation of "{row.optimizer_name}" '
            f'with dimension {row.dimension if hasattr(row, "dimension") else "UNKNOWN"} which raised "{row.error}".'
        )
    err_inds: tp.Set[int] = set(nandf.index)
    output: utils.Selector = df.loc[[i for i in df.index if i not in err_inds], [c for c in df.columns if c != "error"]]
    # cast nans in loss to infinity
    df.loc[np.isnan(df.loss), "loss"] = float("inf")
    #
    assert (
        not output.loc[:, "loss"].isnull().values.any()
    ), "Some nan values remain while there should not be any!"
    output = utils.Selector(output.reset_index(drop=True))
    return output  # type: ignore

class PatternAggregate:
    def __init__(self, pattern: str) -> None:
        self._pattern: str = pattern

    def __call__(self, df: pd.Series) -> str:
        return self._pattern.format(**df.to_dict())

_PARAM_MERGE_PATTERN: str = "{optimizer_name},{parametrization}"

def merge_optimizer_name_pattern(
    df: utils.Selector, pattern: str, merge_parametrization: bool = False, remove_suffix: bool = False
) -> utils.Selector:
    """Merge the optimizer name with other descriptors based on a pattern
    Nothing happens if merge_parametrization is false and pattern is empty string
    """
    if merge_parametrization:
        if pattern:
            raise ValueError(
                "Cannot specify both merge-pattern and merge-parametrization "
                "(merge-parametrization is equivalent to merge-pattern='{optimizer_name},{parametrization}')"
            )
        pattern = _PARAM_MERGE_PATTERN
    if not pattern:
        return df
    df = df.copy()
    okey: str = "optimizer_name"
    elements: tp.List[str] = [tup[1] for tup in string.Formatter().parse(pattern) if tup[1] is not None]
    assert okey in elements, (
        f"Missing optimizer key {okey!r} in merge pattern.\nEg: "
        + 'pattern="{optimizer_name}_{parametrization}"'
    )
    others: tp.List[str] = [x for x in elements if x != okey]
    aggregate: PatternAggregate = PatternAggregate(pattern)
    sub: utils.Selector = df.loc[:, elements].fillna("")
    if len(sub.unique(others)) > 1:
        for optim in sub.unique(okey):
            inds: tp.List[bool] = sub.loc[:, okey] == optim
            if len(sub.loc[inds, :].unique(others)) > 1:
                df.loc[inds, okey] = sub.loc[inds, elements].agg(aggregate, axis=1)
    if remove_suffix:
        df["optimizer_name"] = df["optimizer_name"].replace(r"[0-9\.\-]*$", "", regex=True)
    return df.drop(columns=others)  # type: ignore

def normalized_losses(df: pd.DataFrame, descriptors: tp.List[str]) -> utils.Selector:
    df = utils.Selector(df.copy())
    cases: tp.List[tp.Tuple[tp.Any, ...]] = df.unique(descriptors)
    if not cases:
        cases = [()]
    # Average normalized plot with everything.
    for case in cases:
        subdf: utils.Selector = df.select_and_drop(**dict(zip(descriptors, case)))
        losses: np.ndarray = np.array(subdf.loc[:, "loss"])
        m: float = min(losses)
        M: float = max(losses[losses < (float("inf") if no_limit else float("1e26"))])
        df.loc[subdf.index, "loss"] = (df.loc[subdf.index, "loss"] - m) / (M - m) if M != m else 1
    return df  # type: ignore

# pylint: disable=too-many-statements,too-many-branches
def create_plots(
    df: pd.DataFrame,
    output_folder: tp.PathLike,
    max_combsize: int = 1,
    xpaxis: str = "budget",
    competencemaps: bool = False,
    nomanyxp: bool = False,
) -> None:
    """Saves all representing plots to the provided folder

    Parameters
    ----------
    df: pd.DataFrame
        the experiment data
    output_folder: PathLike
        path of the folder where the plots should be saved
    max_combsize: int
        maximum number of parameters to fix (combinations) when creating experiment plots
    xpaxis: str
        x-axis for xp plots (either budget or pseudotime)
    """
    assert xpaxis in ["budget", "pseudotime"]
    if "non_proxy_function" in df.columns:
        print("removing non_proxy_function")
        df.drop(columns=["non_proxy_function"], inplace=True)
    df = remove_errors(df)
    df.loc[:, "loss"] = pd.to_numeric(df.loc[:, "loss"])
    if not no_limit:
        loss: pd.Series = pd.to_numeric(df.loc[:, "loss"])
        upper: float = np.max(loss[loss < 1e26])
        df.loc[:, "loss"] = df.loc[:, "loss"].clip(lower=-1e26, upper=upper)
    df = df.loc[:, [x for x in df.columns if not x.startswith("info/")]]
    # Normalization of types.
    for col in df.columns:
        print(" Working on ", col)
        failed_indices: tp.List[tp.Any] = []
        if "max_irr" in col:
            df[col] = df[col].round(decimals=4)
        if col in (
            "budget",
            "num_workers",
            "dimension",
            "useful_dimensions",
            "num_blocks",
            "block_dimension",
            "num_objectives",
        ):
            for _ in range(2):
                try:
                    df[col] = df[col].astype(float).astype(int)
                    print(col, " is converted to int")
                    continue
                except Exception as e1:
                    for i in range(len(df[col])):
                        try:
                            float(df[col][i])
                        except Exception as e2:
                            failed_indices += [i]
                            assert (
                                len(failed_indices) < 100
                            ), f"Fails at row {i+2}, Exceptions: {e1}, {e2}. Failed-indices = {failed_indices}"
                print("Dropping ", failed_indices)
                df.drop(df.index[failed_indices], inplace=True)  #        df.drop(index=i, inplace=True)
                failed_indices = []
        #                    print("We drop index ", i, " for ", col)

        elif col != "loss":
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(r"\.[0]*$", "", regex=True)
            try:
                df.loc[:, col] = pd.to_numeric(df.loc[:, col])
            except:
                pass
    if "num_objectives" in df.columns:
        df = df[df.num_objectives != 0]  # the optimization did not even start
    # If we have a descriptor "instrum_str",
    # we assume that it describes the instrumentation as a string,
    # that we should include the various instrumentations as distinct curves in the same plot.
    # So we concat it at the end of the optimizer name, and we remove "parametrization"
    # from the descriptor.
    if "instrum_str" in set(df.columns):
        df.loc[:, "optimizer_name"] = df.loc[:, "optimizer_name"] + df.loc[:, "instrum_str"]
        df = df.drop(columns="instrum_str")
        df = df.drop(columns="dimension")
        if "parametrization" in set(df.columns):
            df = df.drop(columns="parametrization")
        if "instrumentation" in set(df.columns):
            df = df.drop(columns="instrumentation")
    df = utils.Selector(df.fillna("N-A"))  # remove NaN in non score values
    assert not any("Unnamed: " in x for x in df.columns), f"Remove the unnamed index column:  {df.columns}"
    assert "error " not in df.columns, f"Remove error rows before plotting"
    required: tp.Set[str] = {"optimizer_name", "budget", "loss", "elapsed_time", "elapsed_budget"}
    missing: tp.Set[str] = required - set(df.columns)
    assert not missing, f"Missing fields: {missing}"
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    # check which descriptors do vary
    descriptors: tp.List[str] = sorted(
        set(df.columns) - (required | {"instrum_str", "seed", "pseudotime"})
    )  # all other columns are descriptors
    to_drop: tp.List[str] = [x for x in descriptors if len(df.unique(x)) == 1]
    df = utils.Selector(df.loc[:, [x for x in df.columns if x not in to_drop]])
    # now those should be actual interesting descriptors
    all_descriptors: tp.List[str] = sorted(set(df.columns) - (required | {"instrum_str", "seed", "pseudotime"}))
    print(f"Descriptors: {all_descriptors}")
    print("# Fight plots")
    #
    # fight plot
    # choice of the combination variables to fix
    fight_descriptors: tp.List[str] = all_descriptors + ["budget"]  # budget can be used as a descriptor for fight plots
    combinable: tp.List[str] = [x for x in fight_descriptors if len(df.unique(x)) > 1]  # should be all now
    # We remove descriptors which have only one value for each budget.
    descriptors: tp.List[str] = []
    for d in all_descriptors:
        acceptable: bool = False
        for b in df.budget.unique():
            if len(df.loc[df["budget"] == b][d].unique()) > 1:
                acceptable = True
                break
        if acceptable:
            descriptors += [d]
    num_rows: int = 6

    # For the competence map case we must consider pairs of attributes, hence maxcomb_size >= 2.
    # A competence map shows for each value of each of two attributes which algorithm was best.
    if competencemaps:
        max_combsize = max(max_combsize, 2)
    for fixed in list(
        itertools.chain.from_iterable(
            itertools.combinations(combinable, order) for order in range(max_combsize + 1)
        )
    ):
        orders: tp.List[int] = [len(c) for c in df.unique(fixed)]
        if orders:
            assert min(orders) == max(orders)
            order: int = min(orders)
        else:
            order = 0
        best_algo: tp.List[tp.List[str]] = []
        if competencemaps and order == 2:  # With order 2 we can create a competence map.
            print("\n#trying to competence-map")
            if all(
                [len(c) > 1 for c in df.unique(fixed)]
            ):  # Let us try if data are adapted to competence maps.
                # This is not always the case, as some attribute1/value1 + attribute2/value2 might be empty
                # (typically when attribute1 and attribute2 are correlated).
                try:
                    xindices: tp.List[tp.Any] = sorted(set(c[0] for c in df.unique(fixed)))
                except TypeError:
                    xindices: tp.List[tp.Any] = list(set(c[0] for c in df.unique(fixed)))
                try:
                    yindices: tp.List[tp.Any] = sorted(set(c[1] for c in df.unique(fixed)))
                except TypeError:
                    yindices: tp.List[tp.Any] = list(set(c[1] for c in df.unique(fixed)))
                for _ in range(len(xindices)):
                    best_algo += [[]]
                for i in range(len(xindices)):
                    for _ in range(len(yindices)):
                        best_algo[i] += ["none"]

        # Let us loop over all combinations of variables.
        for case in df.unique(fixed) if fixed else [()]:
            print("\n# new case #", fixed, case)
            casedf: utils.Selector = df.select(**dict(zip(fixed, case)))
            data_df: pd.DataFrame = FightPlotter.winrates_from_selection(
                casedf, fight_descriptors, num_rows=num_rows, num_cols=350
            )
            fplotter: FightPlotter = FightPlotter(data_df)
            # Competence maps: we find out the best algorithm for each attribute1=valuei/attribute2=valuej.
            if order == 2 and competencemaps and best_algo:
                print("\n#storing data for competence-map")
                best_algo[xindices.index(case[0])][yindices.index(case[1])] = fplotter.winrates.index[0]
            # save
            name: str = "fight_" + ",".join("{}{}".format(x, y) for x, y in zip(fixed, case)) + ".png"
            name = "fight_all.png" if name == "fight_.png" else name
            name = compactize(name)
            fullname: str = name
            if name == "fight_all.png":
                with open(str(output_folder / name) + ".cp.txt", "w") as f:
                    f.write(fullname)
                    f.write("ranking:\n")
                    for i, algo in enumerate(data_df.columns[:158]):
                        f.write(f"  algo {i}: {algo}\n")

            if len(name) > 240:
                hashcode: str = hashlib.md5(bytes(name, "utf8")).hexdigest()
                name = re.sub(r"\([^()]*\)", "", name)
                mid: int = 120
                name = name[:mid] + hashcode + name[-mid:]
            fplotter.save(str(output_folder / name), dpi=_DPI)
            # Second version, restricted to cases with all data available.
            data_df = FightPlotter.winrates_from_selection(
                casedf, fight_descriptors, num_rows=num_rows, complete_runs_only=True
            )
            fplotter = FightPlotter(data_df)
            if name == "fight_all.png":
                global pure_algorithms
                pure_algorithms = list(data_df.columns[:])
            if name == "fight_all.png":
                fplotter.save(str(output_folder / "fight_all_pure.png"), dpi=_DPI)
            else:
                fplotter.save(str(output_folder / name) + "_pure.png", dpi=_DPI)
                print(f"# {len(data_df.columns[:])}  {data_df.columns[:]}")
            if order == 2 and competencemaps and best_algo:  # With order 2 we can create a competence map.
                print("\n# Competence map")
                name = "competencemap_" + ",".join("{}".format(x) for x in fixed) + ".tex"
                export_table(str(output_folder / name), xindices, yindices, best_algo)
                print("Competence map data:", fixed, case, best_algo)

    plt.close("all")
    # xp plots: for each experimental setup, we plot curves with budget in x-axis.
    # plot mean loss / budget for each optimizer for 1 context
    print("# Xp plots")
    name_style: NameStyle = NameStyle()  # keep the same style for each algorithm
    cases: tp.List[tp.Tuple[tp.Any, ...]] = df.unique(descriptors)
    if not cases:
        cases = [()]
    # Average normalized plot with everything.
    out_filepath: Path = output_folder / "xpresults_all.png"
    try:
        data: tp.Dict[str, tp.Dict[str, np.ndarray]] = XpPlotter.make_data(df, normalized_loss=True)
        for pure_only in [False, True]:
            xpplotter: XpPlotter = XpPlotter(
                data,
                title=os.path.basename(output_folder),
                name_style=name_style,
                xaxis=xpaxis,
                pure_only=pure_only,
            )
    except Exception as e:
        lower: int = 0
        upper: int = len(df)
        while upper > lower + 1:
            middle: int = (lower + upper) // 2
            small_df: utils.Selector = df.head(middle)
            try:
                print("Testing ", middle)
                _ = XpPlotter.make_data(small_df, normalized_loss=True)
                xpplotter = XpPlotter(
                    data,
                    title=os.path.basename(output_folder),
                    name_style=name_style,
                    xaxis=xpaxis,
                    pure_only=True,
                )
                print("Work with ", middle)
                lower = middle
            except:
                print("Failing with ", middle)
                upper = middle

        assert False, f"Big failure {e} at line {middle}"
    xpplotter.save(out_filepath)
    # Now one xp plot per case.
    for case in cases:
        if nomanyxp:
            continue
        subdf: utils.Selector = df.select_and_drop(**dict(zip(descriptors, case)))
        description: str = ",".join("{}:{}".format(x, y) for x, y in zip(descriptors, case))
        full_description: str = description
        description = compactize(description)
        if len(description) > 280:
            hash_: str = hashlib.md5(bytes(description, "utf8")).hexdigest()
            description = description[:140] + hash_ + description[-140:]
        out_filepath = output_folder / "xpresults{}{}.png".format(
            "_" if description else "", description.replace(":", "")
        )
        txt_out_filepath = output_folder / "xpresults{}{}.leaderboard.txt".format(
            "_" if description else "", description.replace(":", "")
        )
        data = XpPlotter.make_data(subdf)
        try:
            xpplotter = XpPlotter(data, title=description, name_style=name_style, xaxis=xpaxis)
        except Exception as e:  # pylint: disable=broad-except
            warnings.warn(f"Bypassing error in xpplotter:\n{e}", RuntimeWarning)
        else:
            xpplotter.save(out_filepath)
            xpplotter.save_txt(txt_out_filepath, data, full_description)
    plt.close("all")

def gp_sota() -> tp.Dict[str, tp.Tuple[float, float]]:
    gp: tp.Dict[str, tp.Tuple[float, float]] = {}
    gp["CartPole-v1"] = (-500.0, 100000.0)
    gp["Acrobot-v1"] = (83.17, 200000.0)
    gp["MountainCarContinuous-v0"] = (-99.31, 900000.0)
    gp["Pendulum-v0"] = (154.36, 1100000.0)
    gp["InvertedPendulumSwingupBulletEnv-v0"] = (-893.35, 400000.0)
    gp["BipedalWalker-v3"] = (-268.85, 1100000.0)
    gp["BipedalWalkerHardcore-v3"] = (-9.25, 1100000.0)
    gp["HopperBulletEnv-v0"] = (-999.19, 1000000.0)
    gp["InvertedDoublePendulumBulletEnv-v0"] = (-9092.17, 300000.0)
    gp["LunarLanderContinuous-v2"] = (-287.58, 1000000.0)
    return gp

def ceviche_sota() -> tp.Dict[str, tp.Tuple[float, float]]:
    ceviche: tp.Dict[str, tp.Tuple[float, float]] = {}
    # {0: "waveguide-bend", 1: "beam-splitter", 2: "mode-converter", 3: "wdm"}

    # Numbers below can be obtained by:
    # grep LOGPB *.out | sed 's/.*://g' | sort | uniq -c | grep with_budget | awk '{ data[$2,"_",$5] += $7;  num[$2,"_",$5] += 1  } END { for (u in data) { print u, data[u]/num[u], num[u]}   } ' | sort -n  | grep '400 '
    # Also obtained by examples/plot_ceviches.sh
    # After log files have been created by sbatch examples/ceviche.sh
    ceviche["waveguide-bend"] = (0.0681388, 1000000)  # Budget 400
    ceviche["beam-splitter"] = (0.496512, 1000000)
    ceviche["mode-converter"] = (0.181592, 1000000)
    ceviche["wdm"] = (0.982352, 100000)
    # LOGPB0 409600 0.0681388
    # LOGPB1 204800 0.496512
    # LOGPB2 204800 0.181592
    # LOGPB3 51200 0.982352
    #
    # LOGPB0_3200 -0.590207
    # LOGPB1_3200 -0.623696
    # LOGPB2_3200 -0.634207
    # LOGPB3_3200 -0.590554
    # LOGPB0_3200 -0.603663
    # LOGPB1_3200 -0.641013
    # LOGPB2_3200 -0.57415
    # LOGPB3_3200 -0.577576
    return ceviche

class LegendInfo(tp.NamedTuple):
    """Handle for information used to create a legend."""

    x: float
    y: float
    line: tp.Any
    text: str

class XpPlotter:
    """Creates a xp result plot out of the given dataframe: regret with respect to budget for
    each optimizer after averaging on all experiments (it is good practice to use a df
    which is filtered out for one set of input parameters)

    Parameters
    ----------
    optim_vals: dict
        output of the make_data static method, containing all information necessary for plotting
    title: str
        title of the plot
    name_style: dict
        a dict or dict-like object providing a line style for each optimizer name.
        (can be helpful for consistency across plots)
    """

    def __init__(
        self,
        optim_vals: tp.Dict[str, tp.Dict[str, np.ndarray]],
        title: str,
        name_style: tp.Optional[tp.Dict[str, tp.Any]] = None,
        xaxis: str = "budget",
        pure_only: bool = False,
    ) -> None:
        if name_style is None:
            name_style = NameStyle()
        upperbound: float = max(
            np.max(vals["loss"]) for vals in optim_vals.values() if np.max(vals["loss"]) < np.inf
        )
        for optim, vals in optim_vals.items():
            if optim.lower() in ["stupid", "idiot"] or optim in ["Zero", "StupidRandom"]:
                upperbound = min(upperbound, np.max(vals["loss"]))
        # plot from best to worst
        lowerbound: float = np.inf
        sorted_optimizers: tp.List[str] = sorted(optim_vals, key=lambda x: optim_vals[x]["loss"][-1], reverse=True)
        if pure_only:
            assert len(pure_algorithms) > 0
            sorted_optimizers = [
                o for o in sorted_optimizers if o + " " in [p[: (len(o) + 1)] for p in pure_algorithms]
            ]
        with open(("rnk__" if not pure_only else "rnkpure__") + str(title) + ".cp.txt", "w") as f:
            f.write(compactize(title))
            f.write("ranking:\n")
            for i, algo in reversed(list(enumerate(sorted_optimizers))):
                f.write(f"  algo {i}: {algo} (x)\n")
            # print(sorted_optimizers, " merged with ", pure_algorithms)
            # print("Leads to ", sorted_optimizers)
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        # use log plot? yes, if no negative value
        logplot: bool = not any(
            x <= 0 or x > 10**8 for ov in optim_vals.values() for x in ov["loss"]
        )  # if x < np.inf)
        if logplot:
            self._ax.set_yscale("log")
            for ov in optim_vals.values():
                if ov["loss"].size:
                    ov["loss"] = np.maximum(1e-30, ov["loss"])
        # other setups
        self._ax.autoscale(enable=False)
        self._ax.set_xscale("log")
        self._ax.set_xlabel(xaxis)
        self._ax.set_ylabel("loss")
        self._ax.grid(True, which="both")
        self._overlays: tp.List[tp.Any] = []
        legend_infos: tp.List[LegendInfo] = []
        title_addendum: str = f"({len(sorted_optimizers)} algos)"
        for optim_name in (
            sorted_optimizers[:1] + sorted_optimizers[-35:]
            if len(sorted_optimizers) > 35
            else sorted_optimizers
        ):
            vals = optim_vals[optim_name]

            indices: np.ndarray = np.where(vals["num_eval"] > 0)
            lowerbound = min(lowerbound, np.min(vals["loss"]))
            # We here add some state of the art results.
            # This adds a cross on figures, x-axis = budget and y-axis = loss.
            for sota_name, sota in [("GP", gp_sota()), ("ceviche", ceviche_sota())]:
                for k in sota.keys():
                    if k in title:
                        th: float = sota[k][0]  # loss of proposed solution.
                        cost: float = sota[k][1]  # Computational cost for the proposed result.
                        title_addendum = f"({sota_name}:{th})"
                        lowerbound = min(lowerbound, th, 0.9 * th, 1.1 * th)
                        plt.plot(  # Horizontal line at the obtained GP cost.
                            vals[xaxis][indices],
                            th + 0 * vals["loss"][indices],
                            name_style[optim_name],
                            label=sota_name,
                        )
                        plt.plot(  # Vertical line, showing the budget of the GP solution.
                            [cost] * 3,
                            [
                                min(vals["loss"][indices]),
                                sum(vals["loss"][indices]) / len(indices),
                                max(vals["loss"][indices]),
                            ],
                            name_style[optim_name],
                            label=sota_name,
                        )
            line = plt.plot(vals[xaxis], vals["loss"], name_style[optim_name], label=optim_name)
            # confidence lines
            for conf in self._get_confidence_arrays(vals, log=logplot):
                plt.plot(vals[xaxis], conf, name_style[optim_name], label=optim_name, alpha=0.1)
            text: str = "{} ({:.3g} <{:.3g}>)".format(
                optim_name,
                vals["loss"][-1],
                vals["loss"][-2] if len(vals["loss"]) > 1 else float("nan"),
            )
            if vals[xaxis].size:
                legend_infos.append(LegendInfo(vals[xaxis][-1], vals["loss"][-1], line, text))
        if not (np.isnan(upperbound) or np.isinf(upperbound)):
            upperbound_up: float = upperbound
            if not (np.isnan(lowerbound) or np.isinf(lowerbound)):
                self._ax.set_ylim(bottom=lowerbound)
                upperbound_up += 0.02 * (upperbound - lowerbound)
                if logplot:
                    upperbound_up = 10 ** (
                        np.log10(upperbound) + 0.02 * (np.log10(upperbound) - np.log10(lowerbound))
                    )
            self._ax.set_ylim(top=upperbound_up)
        all_x: tp.List[float] = [v for vals in optim_vals.values() for v in vals[xaxis]]
        try:
           