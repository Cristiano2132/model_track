from typing import Dict, List, Optional
import pandas as pd


class CategoryMapper:
    """
    Creates and stores category grouping mappings based on WOE tables.

    Supports:
    - multiple features
    - numeric ordered categories
    - interval-like categories
    - manual override via set()
    """

    def __init__(self):
        self._mappings: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_map(
        self,
        df: pd.DataFrame,
        feature_name: str,
        category_col: str,
        groups: List[List[int]],
        ordered: bool = False,
    ) -> Dict[str, str]:
        """
        Create a category mapping for a feature based on a global WOE table.

        Parameters
        ----------
        df : pd.DataFrame
            Global WOE table.
        feature_name : str
            Name of the feature being mapped.
        category_col : str
            Column containing category labels.
        groups : List[List[int]]
            Indices of rows to be grouped together.
        ordered : bool
            Whether categories are ordered (numeric-like).
        """
        categories = (
            df[category_col]
            .dropna()
            .astype(str)
            .tolist()
        )

        # remove TOTAL row if present
        categories = [c for c in categories if c != "__TOTAL__"]

        mapping: Dict[str, str] = {}

        # prepare numeric domain if needed
        numeric_domain = None
        if ordered and self._all_numeric(categories):
            numeric_domain = sorted(map(float, categories))
            global_min = numeric_domain[0]
            global_max = numeric_domain[-1]
        else:
            global_min = global_max = None

        for group in groups:
            group_cats = [categories[i] for i in group]

            if ordered and numeric_domain is not None:
                new_label = self._numeric_ordered_label(
                    group_cats,
                    global_min=global_min,
                    global_max=global_max,
                )
            elif ordered and self._is_interval_like(group_cats):
                if self._check_continuous_intervals(categories, group_cats):
                    new_label = self._merge_interval_labels(group_cats)
                else:
                    new_label = self._concat_label(group_cats)
            else:
                new_label = self._concat_label(group_cats)

            for cat in group_cats:
                mapping[cat] = new_label

        # identity mapping for non-grouped categories
        for cat in categories:
            if cat not in mapping:
                mapping[cat] = cat

        self._mappings[feature_name] = mapping
        return mapping

    def get(self, feature_name: Optional[str] = None):
        """
        Retrieve mappings.

        - If feature_name is provided, returns mapping for that feature.
        - Otherwise, returns all mappings.
        """
        if feature_name:
            return self._mappings.get(feature_name, {})
        return self._mappings

    def set(self, feature_name: str, mapping: Dict[str, str]):
        """
        Manually override or define a mapping for a feature.
        """
        self._mappings[feature_name] = mapping

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _concat_label(self, categories: List[str]) -> str:
        return " | ".join(categories)

    def _numeric_ordered_label(
        self,
        categories: List[str],
        global_min: float,
        global_max: float,
    ) -> str:
        values = sorted(map(float, categories))

        # lower edge
        if values[0] == global_min:
            return f"<={int(values[-1])}"

        # upper edge
        if values[-1] == global_max:
            return f">={int(values[0])}"

        # middle group
        return f"[{int(values[0])}, {int(values[-1])}]"

    def _all_numeric(self, categories: List[str]) -> bool:
        try:
            for c in categories:
                float(c)
            return True
        except ValueError:
            return False
    
    def _is_interval_like(self, categories: List[str]) -> bool:
        padroes = ['(', ')', '[', ']', '<=', '>=', '<', '>']
        for c in categories:
            if not any(p in c for p in padroes):
                return False
        return True

    def _merge_interval_labels(self, categories: List[str]) -> str:
        lowers = []
        uppers = []
        has_minus_inf = False
        has_plus_inf = False

        for c in categories:
            c = c.strip()

            if c.startswith("<="):
                has_minus_inf = True
                uppers.append(float(c.replace("<=", "").strip()))

            elif c.startswith(">="):
                has_plus_inf = True
                lowers.append(float(c.replace(">=", "").strip()))

            else:
                # interval like (a,b]
                left, right = c[1:-1].split(",")
                lowers.append(float(left.strip()))
                uppers.append(float(right.strip()))

        min_lower = min(lowers) if lowers else None
        max_upper = max(uppers) if uppers else None

        # (-inf, x]
        if has_minus_inf and not has_plus_inf:
            return f"<={int(max_upper)}"

        # [x, +inf)
        if has_plus_inf and not has_minus_inf:
            return f">={int(min_lower)}"

        # fully bounded interval
        return f"({int(min_lower)},{int(max_upper)}]"

    def _parse_interval(self, label: str):
        label = label.strip()

        if label.startswith("<="):
            return (-float("inf"), float(label.replace("<=", "").strip()))

        if label.startswith(">="):
            return (float(label.replace(">=", "").strip()), float("inf"))

        # (a,b]
        left, right = label[1:-1].split(",")
        return (float(left.strip()), float(right.strip()))
    
    def _check_continuous_intervals(self, categories: list[str], group_cats: list[str]) -> bool:
        group_intervals = [self._parse_interval(c) for c in group_cats]
        group_intervals = sorted(group_intervals, key=lambda x: x[0])
        for i in range(len(group_intervals) - 1):
            current_upper = group_intervals[i][1]
            next_lower = group_intervals[i + 1][0]
            if current_upper != next_lower:
                return False
        return True