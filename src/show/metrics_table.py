import pandas as pd


class MetricsTable:
    def __init__(self, methods_metrics):
        """
        Initializes the MetricsTable with clustering evaluation results.
        :param methods_metrics: Dictionary containing evaluation metrics per method and semantic map.
        """
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.expand_frame_repr", False)
        self.methods_metrics = methods_metrics
        self.df = self._create_dataframe()

    def _create_dataframe(self):
        """
        Converts the nested dictionary of metrics into a Pandas DataFrame
        without grouping/aggregation (one row per method and semantic map).
        """
        data = []
        for method, semantic_maps in self.methods_metrics.items():
            for semantic_map, metrics in semantic_maps.items():
                # Determine the dataset from the semantic_map name
                if semantic_map.startswith("scannet"):
                    dataset = "scannet"
                elif semantic_map.startswith("scenenn"):
                    dataset = "scenenn"
                else:
                    dataset = "unknown"  # or handle differently if desired

                data.append({
                    "Method": method,
                    "Semantic Map": semantic_map,
                    "Dataset": dataset,
                    "ARI": metrics["ARI"],
                    "NMI": metrics["NMI"],
                    "V-Measure": metrics["V-Measure"],
                    "FMI": metrics["FMI"],
                })

        return pd.DataFrame(data)

    def display(self, group_by=None):
        """
        Displays the DataFrame. If 'group_by' is provided,
        groups by those columns and shows the mean of numeric metrics.

        :param group_by: list of column names to group by (e.g. ["Method", "Dataset"])
        """
        print("*" * 100)
        print("Displaying the metrics:")

        if group_by:
            numeric_cols = ["ARI", "NMI", "V-Measure", "FMI"]
            grouped_df = (
                self.df
                .groupby(group_by, as_index=False)[numeric_cols]
                .mean()
            )
            print(grouped_df)
        else:
            print(self.df)

    def display_best(self, num_best, metric, group_by=None):
        """
        Displays the top N (num_best) rows based on a specific metric.
        If 'group_by' is provided, aggregates first by those columns
        before sorting and displaying the top rows.

        :param num_best: Number of best rows to display.
        :param metric: Metric to sort by ("ARI", "NMI", "V-Measure", "FMI").
        :param group_by: (Optional) List of column names to group by before sorting.
        """
        print("*" * 100)
        if metric not in ["ARI", "NMI", "V-Measure", "FMI"]:
            raise ValueError(
                "Invalid metric. Choose from 'ARI', 'NMI', 'V-Measure', 'FMI'."
            )

        if group_by:
            numeric_cols = ["ARI", "NMI", "V-Measure", "FMI"]
            df_to_sort = (
                self.df
                .groupby(group_by, as_index=False)[numeric_cols]
                .mean()
            )
        else:
            df_to_sort = self.df

        print(f"Best {num_best} entries in terms of '{metric}'"
              + (f" (grouped by {group_by})" if group_by else ""))

        best_df = df_to_sort.sort_values(
            by=metric, ascending=False).head(num_best)
        print(best_df)

    def display_worst(self, num_worst, metric, group_by=None):
        """
        Displays the bottom N (num_worst) rows based on a specific metric.
        If 'group_by' is provided, aggregates first by those columns
        before sorting and displaying the bottom rows.

        :param num_worst: Number of worst rows to display.
        :param metric: Metric to sort by ("ARI", "NMI", "V-Measure", "FMI").
        :param group_by: (Optional) List of column names to group by before sorting.
        """
        print("*" * 100)
        if metric not in ["ARI", "NMI", "V-Measure", "FMI"]:
            raise ValueError(
                "Invalid metric. Choose from 'ARI', 'NMI', 'V-Measure', 'FMI'."
            )

        if group_by:
            numeric_cols = ["ARI", "NMI", "V-Measure", "FMI"]
            df_to_sort = (
                self.df
                .groupby(group_by, as_index=False)[numeric_cols]
                .mean()
            )
        else:
            df_to_sort = self.df

        print(f"Worst {num_worst} entries in terms of '{metric}'"
              + (f" (grouped by {group_by})" if group_by else ""))

        worst_df = df_to_sort.sort_values(
            by=metric, ascending=True).head(num_worst)
        print(worst_df)

    def filter_methods(self, search_str, group_by=None):
        """
        Filters and displays the rows (or grouped rows) where the 'Method' column 
        contains the given search string, ordering the results from best to worst 
        based on the mean of the metrics.

        :param search_str: Substring to search for in the 'Method' column.
        :param group_by: (Optional) List of column names to group by before sorting.
        """
        print("*" * 100)
        print(f"Methods containing '{search_str}'" +
              (f" (grouped by {group_by})" if group_by else ""))

        # First, filter rows that match the search string
        filtered_df = self.df[
            self.df["Method"].str.contains(search_str, case=False, na=False)
        ].copy()

        # If group_by is not None, group and compute averages
        if group_by:
            numeric_cols = ["ARI", "NMI", "V-Measure", "FMI"]
            grouped_df = (
                filtered_df.groupby(group_by, as_index=False)[numeric_cols]
                           .mean()
            )
            # Calculate mean of metrics for sorting
            grouped_df["Mean Score"] = grouped_df[numeric_cols].mean(axis=1)
            # Sort descending by Mean Score
            grouped_df = grouped_df.sort_values(
                by="Mean Score", ascending=False)
            print(grouped_df.drop(columns=["Mean Score"]))
        else:
            # If no grouping, just compute a per-row Mean Score and sort
            filtered_df["Mean Score"] = filtered_df[[
                "ARI", "NMI", "V-Measure", "FMI"]].mean(axis=1)
            filtered_df = filtered_df.sort_values(
                by="Mean Score", ascending=False)
            print(filtered_df.drop(columns=["Mean Score"]))
