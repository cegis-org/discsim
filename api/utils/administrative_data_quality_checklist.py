import numpy as np
import pandas as pd
from itertools import combinations
from typing import Union, List, Tuple, Optional, Dict
from scipy.stats import binom

def run_preliminary_tests(df: pd.DataFrame) -> Dict[str, Union[int, str, List[str]]]:
    """
    Run preliminary tests on the uploaded dataset.

    Args:
    df (pd.DataFrame): The input dataframe

    Returns:
    Dict[str, Union[int, str, List[str]]]: A dictionary containing test results
    """
    results = {"status": 0, "error_code": None, "message": "Success", "warnings": []}

    # Check if the dataset has more than one column
    if df.shape[1] == 1:
        results["status"] = 2
        results["error_code"] = 1
        results["message"] = "The uploaded dataset only has 1 column"
        return results

    # Check if the dataset has at least two rows
    if df.shape[0] < 2:
        results["status"] = 2
        results["error_code"] = 2
        results["message"] = "The uploaded dataset has less than 2 rows"
        return results

    # Check for missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    if missing_columns:
        results["warnings"].append(
            f"The following columns have missing values: {', '.join(missing_columns)}"
        )

    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        results["warnings"].append(
            f"The dataset contains {duplicate_count} duplicate rows"
        )

    return results


def findUniqueIDs(data: List[Dict]) -> List[Dict[str, Union[List[str], int]]]:
    """
    Find unique identifiers in the given dataset, excluding combinations with pre-existing unique IDs.

    This function takes a dataset as input and returns a list of columns
    or combinations that can be used as unique identifiers, sorted by length
    and number of numeric data types.

    Args:
    data (List[Dict]): The input dataset as a list of dictionaries.

    Returns:
    List[Dict[str, Union[List[str], int]]]: A list of dictionaries containing unique IDs
    and their numeric datatype count, sorted by length and number of numeric data types.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame(data)
    uniqueIDcols = []
    single_unique_cols = set()

    def get_column_with_dtype(column: str) -> str:
        """Get column name with its datatype in brackets."""
        dtype = df[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            return f"{column} (numeric)"
        elif pd.api.types.is_string_dtype(dtype):
            return f"{column} (string)"
        else:
            return f"{column} (other)"

    def count_numeric_datatypes(columns: List[str]) -> int:
        """Count the number of numeric datatypes in given columns."""
        return sum(df[col].dtype in ["int64", "float64"] for col in columns)

    # Check individual columns first
    for col in df.columns:
        if df[col].nunique() == len(df):
            uniqueIDcols.append(
                {"UniqueID": [get_column_with_dtype(col)], "Numeric_DataTypes": count_numeric_datatypes([col])}
            )
            single_unique_cols.add(col)

    # Check combinations of 2 and 3 columns, excluding those with pre-existing unique IDs
    for r in range(2, 4):
        for combo in combinations(df.columns, r):
            if not any(col in single_unique_cols for col in combo):
                if df.groupby(list(combo)).ngroups == len(df):
                    num_numeric_datatypes = count_numeric_datatypes(combo)
                    uniqueIDcols.append(
                        {
                            "UniqueID": [get_column_with_dtype(col) for col in combo],
                            "Numeric_DataTypes": num_numeric_datatypes,
                        }
                    )

    # Sort the results by length of UniqueID and number of numeric datatypes
    return sorted(
        uniqueIDcols, key=lambda x: (len(x["UniqueID"]), -x["Numeric_DataTypes"])
    )


def uniqueIDcheck(data: List[Dict], colsList: List[str]) -> Tuple[str, bool]:
    """
    Check if selected columns form a unique identifier in the dataset.

    This function verifies if the selected column(s) can serve as a unique identifier
    for the given dataset.

    Args:
    data (List[Dict]): The input dataset as a list of dictionaries.
    colsList (List[str]): List of column names to check for uniqueness.

    Returns:
    Tuple[str, bool]: A tuple containing the result message and a boolean indicating
    whether the selected columns can work as a unique ID.
    """
    # Input validation
    if not colsList:
        return "No columns selected", False

    if len(colsList) > 4:
        return "You have selected more than 4 columns", False

    # Convert input data to DataFrame
    df = pd.DataFrame(data)

    # Check if all selected columns are in the dataframe
    if not set(colsList).issubset(df.columns):
        return "Selected Column(s) are not in the dataframe uploaded", False

    # Check for uniqueness
    if len(colsList) == 1:
        is_unique = df[colsList[0]].is_unique
    else:
        is_unique = df.duplicated(subset=colsList).sum() == 0

    # Return result
    if is_unique:
        return "Selected column(s) can work as unique ID", True
    else:
        return "Selected column(s) cannot work as unique ID", False


def dropExportDuplicates(
    df1: Union[pd.DataFrame, str],
    uidCol: Union[str, List[str]],
    keptRow: str = "first",
    export: bool = True,
    chunksize: Optional[int] = None,
) -> Tuple[List[Dict], Optional[List[Dict]]]:
    if isinstance(df1, str):
        if chunksize:
            return process_in_chunks(df1, uidCol, keptRow, export, chunksize)
        else:
            df1 = pd.read_csv(df1, keep_default_na=False, na_values=[""])

    df1 = df1.replace([np.inf, -np.inf], np.nan)

    keep_param = False if keptRow.lower() == "none" else keptRow.lower()

    is_duplicate = df1.duplicated(subset=uidCol, keep=False)
    df_unique = df1[~df1.duplicated(subset=uidCol, keep=keep_param)]
    df_dupl = df1[is_duplicate] if export else None

    def safe_convert(df):
        return df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(
            "records"
        )

    unique_rows = safe_convert(df_unique)
    duplicate_rows = safe_convert(df_dupl) if df_dupl is not None else None

    return unique_rows, duplicate_rows


def process_in_chunks(
    file_path: str,
    uidCol: Union[str, List[str]],
    keptRow: str,
    export: bool,
    chunksize: int,
) -> Tuple[List[Dict], Optional[List[Dict]]]:
    unique_rows = []
    duplicate_rows = []

    keep_param = False if keptRow.lower() == "none" else keptRow.lower()

    def safe_convert(df):
        return df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(
            "records"
        )

    for chunk in pd.read_csv(
        file_path, chunksize=chunksize, keep_default_na=False, na_values=[""]
    ):
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        is_duplicate = chunk.duplicated(subset=uidCol, keep=False)
        unique_rows.extend(
            safe_convert(chunk[~chunk.duplicated(subset=uidCol, keep=keep_param)])
        )
        if export:
            duplicate_rows.extend(safe_convert(chunk[is_duplicate]))

    return unique_rows, duplicate_rows if export else None


def missingEntries(df: pd.DataFrame, colName: str) -> Tuple[int, Optional[float], int]:
    missingCount = df[colName].isna().sum()
    totalCount = len(df)
    if totalCount == 0:
        return missingCount, None, totalCount
    missingPercentage = 100 * missingCount / totalCount
    if np.isnan(missingPercentage) or np.isinf(missingPercentage):
        return missingCount, None, totalCount
    return missingCount, float(missingPercentage), totalCount


def missingEntriesGrouped(
    df: pd.DataFrame, colName: str, catColumn: str
) -> Dict[str, Tuple[int, float]]:
    """
    Calculate missing entries grouped by a categorical variable.

    Args:
    df (pd.DataFrame): Input dataframe
    colName (str): Name of the column to check for missing entries
    catColumn (str): Name of the categorical column to group by

    Returns:
    Dict[str, Tuple[int, float]]: Dictionary with group names as keys and (missing count, missing percentage) as values
    """
    return df.groupby(catColumn).apply(lambda x: missingEntries(x, colName)).to_dict()


def missingEntriesFiltered(
    df: pd.DataFrame, colName: str, catColumn: str, catValue: str
) -> Tuple[int, float]:
    """
    Calculate missing entries filtered by a categorical variable.

    Args:
    df (pd.DataFrame): Input dataframe
    colName (str): Name of the column to check for missing entries
    catColumn (str): Name of the categorical column to filter by
    catValue (str): Value of the categorical column to filter on

    Returns:
    Tuple[int, float]: (missing count, missing percentage) for the filtered data
    """
    filtered_df = df[df[catColumn] == catValue]
    return missingEntries(filtered_df, colName)


def analyze_missing_entries(
    df: pd.DataFrame,
    colName: str,
    groupBy: Optional[str] = None,
    filterBy: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[
    Tuple[int, Optional[float], int],
    Dict[str, Tuple[int, Optional[float], int]]
]]:
    """
    Analyze missing entries in a dataframe column with optional grouping and filtering.

    Args:
    df (pd.DataFrame): Input dataframe
    colName (str): Name of the column to check for missing entries
    groupBy (Optional[str]): Name of the categorical column to group by (default: None)
    filterBy (Optional[Dict[str, str]]): Dictionary with column name as key and value to filter on (default: None)

    Returns:
    Dict[str, Union[Tuple[int, float], Dict[str, Tuple[int, float]]]]: Dictionary with analysis results
    """
    result = {}

    if filterBy:
        for col, value in filterBy.items():
            if df[col].dtype in ["int64", "float64"]:
                df = df[df[col] == float(value)]
            else:
                df = df[df[col].astype(str) == str(value)]
        result["filtered"] = True
        if df.empty:
            raise ValueError(f"No data found for filter: {filterBy}")
    else:
        result["filtered"] = False

    if df.empty:
        raise ValueError("The resulting dataframe is empty")

    if groupBy:
        result["grouped"] = True
        result["analysis"] = missingEntriesGrouped(df, colName, groupBy)
    else:
        result["grouped"] = False
        result["analysis"] = missingEntries(df, colName)

    return result


def zeroEntries(df: pd.DataFrame, colName: str) -> Tuple[int, float , int]:
    """
    Calculate the number and percentage of zero entries in a column.

    Args:
    df (pd.DataFrame): Input dataframe
    colName (str): Name of the column to check for zero entries

    Returns:
    Tuple[int, float]: (zero count, zero percentage)
    """
    totalRows = len(df)
    if df[colName].dtype not in ["int64", "float64"]:
        return 0, 0.0 , totalRows
    zeroCount = (df[colName] == 0).sum()
    zeroPercentage = (zeroCount / totalRows * 100) if totalRows > 0 else 0.0
    return zeroCount, zeroPercentage, totalRows


def zeroEntriesGrouped(
    df: pd.DataFrame, colName: str, catColumn: str
) -> Dict[str, Tuple[int, float, int]]:
    """
    Calculate zero entries grouped by a categorical variable.

    Args:
    df (pd.DataFrame): Input dataframe
    colName (str): Name of the column to check for zero entries
    catColumn (str): Name of the categorical column to group by

    Returns:
    Dict[str, Tuple[int, float]]: Dictionary with group names as keys and (zero count, zero percentage) as values
    """
    if df[colName].dtype not in ["int64", "float64"]:
        return {group: (0, 0.0, len(group_df)) for group,group_df in df[catColumn].unique()}
    return df.groupby(catColumn).apply(lambda x: zeroEntries(x, colName)).to_dict()


def zeroEntriesFiltered(
    df: pd.DataFrame, colName: str, catColumn: str, catValue: str
) -> Tuple[int, float ,int]:
    """
    Calculate zero entries filtered by a categorical variable.

    Args:
    df (pd.DataFrame): Input dataframe
    colName (str): Name of the column to check for zero entries
    catColumn (str): Name of the categorical column to filter by
    catValue (str): Value of the categorical column to filter on

    Returns:
    Tuple[int, float]: (zero count, zero percentage) for the filtered data
    """
    filtered_df = df[df[catColumn] == catValue]
    return zeroEntries(filtered_df, colName)


def analyze_zero_entries(
    df: pd.DataFrame,
    colName: str,
    groupBy: Optional[str] = None,
    filterBy: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[
    Tuple[int, Optional[float], int],
    Dict[str, Tuple[int, Optional[float], int]]
]]:
    """
    Analyze zero entries in a dataframe column with optional grouping and filtering.

    Args:
    df (pd.DataFrame): Input dataframe
    colName (str): Name of the column to check for zero entries
    groupBy (Optional[str]): Name of the categorical column to group by (default: None)
    filterBy (Optional[Dict[str, str]]): Dictionary with column name as key and value to filter on (default: None)

    Returns:
    Dict[str, Union[Tuple[int, float], Dict[str, Tuple[int, float]]]]: Dictionary with analysis results
    """
    result = {}

    if filterBy:
        for col, value in filterBy.items():
            df = df[df[col] == value]
        result["filtered"] = True
    else:
        result["filtered"] = False

    if groupBy:
        result["grouped"] = True
        result["analysis"] = zeroEntriesGrouped(df, colName, groupBy)
    else:
        result["grouped"] = False
        result["analysis"] = zeroEntries(df, colName)

    # Include rows with zero entries
    zero_rows = df[df[colName] == 0]
    result["zero_entries_table"] = zero_rows.reset_index().to_dict(orient="records")

    return result


def apply_invalid_condition(series: pd.Series, conditions: List[Dict], df: pd.DataFrame) -> Dict[str, pd.Series]:
    if is_numeric_column(series):
        return apply_numeric_conditions(series, conditions,df)
    elif is_string_column(series):
        return apply_string_conditions(series, conditions, df)
    elif is_datetime_column(series):
        return apply_datetime_conditions(series, conditions)
    else:
        raise ValueError("Unsupported column type for invalid condition.")

def apply_numeric_conditions(series: pd.Series, conditions: List[Dict], df: pd.DataFrame) -> Dict[str, pd.Series]:
    result = {}
    for cond in conditions:
        op = cond['operation']
        val = cond['value']
        label = cond.get('label', 'Invalid')

        if op == "<":
            result[label] = series < val
        elif op == "<=":
            result[label] = series <= val
        elif op == ">":
            result[label] = series > val
        elif op == ">=":
            result[label] = series >= val
        elif op == "==":
            result[label] = series == val
        elif op == "!=":
            result[label] = series != val
        elif op == "between":
            if isinstance(val, (list, tuple)) and len(val) == 2:
                lower, upper = val
                result[label] = series.between(lower, upper, inclusive='both')
            else:
                raise ValueError(f"'between' operation expects a tuple/list with 2 values. Got: {val}")
        elif op == "Compare Equals":
            compare_col = pd.to_numeric(df[val], errors='coerce')
            result[label] = series == compare_col
        elif op == "Compare Not Equals":
            compare_col = pd.to_numeric(df[val], errors='coerce')
            result[label] = series != compare_col
        else:
            raise ValueError(f"Invalid operation: {op}")
        
    return result

def apply_string_conditions(series: pd.Series, conditions: List[Dict], df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Apply multiple string invalid conditions on a Series.

    Returns a dictionary {criteria_label: mask_series}.
    """
    masks = {}

    for cond in conditions:
        operation = cond["operation"]
        value = cond["value"]
        label = cond["label"]

        if operation == "Contains":
            mask = series.str.contains(value, na=False)
        elif operation == "Does not contain":
            mask = ~series.str.contains(value, na=False)
        elif operation == "Equals":
            mask = series == value
        elif operation == "Not equals":
            mask = series != value
        elif operation == "Compare Equals":
            mask = series.fillna("").astype(str) == df[value].fillna("").astype(str)
        elif operation == "Compare Not Equals":
            mask = series.fillna("").astype(str) != df[value].fillna("").astype(str)
        else:
            raise ValueError(f"Invalid string operation: {operation}")

        masks[label] = mask

    return masks



def apply_datetime_conditions(series: pd.Series, conditions: List[Dict]) -> Dict[str, pd.Series]:
    """
    Apply multiple datetime invalid conditions on a Series.

    Returns a dictionary {criteria_label: mask_series}.
    """
    # Ensure the series is in datetime format
    series = pd.to_datetime(series, errors="coerce")

    masks = {}

    for cond in conditions:
        label = cond["label"]
        start_date_str, end_date_str = cond["value"]

        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        # Mask for values between start_date (exclusive) and end_date (inclusive)
        mask = (series > start_date) & (series <= end_date)

        masks[label] = mask

    return masks



def is_numeric_column(series):
    return (
        pd.api.types.is_numeric_dtype(series)
        or series.dtype == "object"
        and series.str.isnumeric().all()
    )


def is_string_column(series):
    return pd.api.types.is_string_dtype(series)


def is_datetime_column(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Try to parse the first non-null value as a date
    first_valid = series.first_valid_index()
    if first_valid is not None:
        try:
            pd.to_datetime(series[first_valid])
            return True
        except:
            pass

    return False


def parse_dates(df, column):
    try:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    except:
        pass
    return df


def get_numeric_operations():
    return ["<", "<=", ">", ">=", "==", "!="]


def get_string_operations():
    return ["Contains", "Does not contain", "Equals", "Not equals"]

def parse_invalid_condition(condition_input) -> Tuple[str, str, str]:
    if isinstance(condition_input, tuple) and len(condition_input) == 3:
        return condition_input

    if isinstance(condition_input, list) and len(condition_input) == 3:
        return tuple(condition_input)

    if isinstance(condition_input, str):
        parts = condition_input.strip().split(maxsplit=2)
        if len(parts) < 3:
            raise ValueError("Invalid condition string must be in the format: '<operation> <value> <label>'")
        operation, value, label = parts[0], parts[1], parts[2]
        return operation, value, label

    raise ValueError("Invalid condition must be a string, list, or tuple of 3 elements.")



def indicatorFillRate(
    df: pd.DataFrame,
    colName: str,
    invalid_conditions: Optional[List[Dict]] = None,
    include_zero_as_separate_category: bool = True,
) -> pd.DataFrame:
    total = len(df)
    series = df[colName]

    missing = series.isnull()

    if is_numeric_column(series):
        series = pd.to_numeric(series, errors="coerce")
    elif is_string_column(series) or is_datetime_column(series):
        pass  # No conversion needed
    else:
        raise ValueError(f"Unsupported data type for column: {colName}")

    missing = series.isnull()
    zero = (series == 0) if is_numeric_column(series) else pd.Series(False, index=series.index)
    invalid_masks = apply_invalid_condition(series, invalid_conditions or [], df)
    combined_invalid = pd.Series(False, index=series.index)
    rows = [{"Category": "Missing", "Number of observations": missing.sum()}]
    if include_zero_as_separate_category:
        rows.append({"Category": "Zero", "Number of observations": zero.sum()})
    for label, mask in invalid_masks.items():
            cleaned_mask = mask & ~missing
            rows.append({
                "Category": label,
                "Number of observations": cleaned_mask.sum()
            })
            combined_invalid |= cleaned_mask
    
    valid = ~(missing | zero | combined_invalid)
    rows.append({"Category": "Valid", "Number of observations": valid.sum()})

    result_df = pd.DataFrame(rows)
    result_df["Percentage of observations"] = (result_df["Number of observations"] / total * 100).round(1)
    return result_df


def indicatorFillRateGrouped(
    df: pd.DataFrame,
    colName: str,
    catColumn: str,
    invalid_conditions: Optional[List[Tuple[str, Union[str, float], str]]] = None,
    include_zero_as_separate_category: bool = True,
) -> Dict[str, pd.DataFrame]:
    return {
        name: indicatorFillRate(
            group, colName, invalid_conditions, include_zero_as_separate_category
        )
        for name, group in df.groupby(catColumn)
    }

def indicatorFillRateFiltered(
    df: pd.DataFrame,
    colName: str,
    catColumn: str,
    catValue: str,
    invalid_condition: Optional[Union[str, Tuple[str, str, str]]] = None,
) -> pd.DataFrame:
    return indicatorFillRate(df[df[catColumn] == catValue], colName, invalid_condition)


def analyze_indicator_fill_rate(
    df: pd.DataFrame,
    colName: str,
    groupBy: Optional[str] = None,
    filterBy: Optional[Dict[str, str]] = None,
    invalid_conditions: Optional[List[Dict]] = None,
    include_zero_as_separate_category: bool = True,
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    result = {}

    result["total"] = len(df)

    if filterBy:
        for col, value in filterBy.items():
            df = df[df[col] == value]
        result["total"] = len(df)
        result["filtered"] = True
    else:
        result["filtered"] = False

    def get_detailed_data(
        data: pd.DataFrame,
        colName: str,
        invalid_conditions: Optional[List[Dict]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Get detailed breakdown of data based on missing, zero, invalid, and valid entries.

        Args:
            data (pd.DataFrame): Input dataframe.
            colName (str): Column name to analyze.
            invalid_conditions (List[Dict], optional): List of invalid conditions.

        Returns:
            Dict[str, List[Dict]]: Mapping category -> list of record dictionaries.
        """
        series = data[colName]

        # Handle type conversion
        if is_numeric_column(series):
            series = pd.to_numeric(series, errors="coerce")
        elif is_datetime_column(series):
            series = pd.to_datetime(series, errors="coerce")
        elif not is_string_column(series):
            raise ValueError(f"Unsupported data type for column: {colName}")

        # Build base masks
        missing_mask = series.isnull()
        zero_mask = (series == 0) if is_numeric_column(series) else pd.Series(False, index=series.index)

        # Apply invalid conditions
        invalid_masks = apply_invalid_condition(series, invalid_conditions or [],df)
        combined_invalid_mask = pd.Series(False, index=series.index)

        detailed = {}

        # Collect invalid entries
        for label, mask in invalid_masks.items():
            clean_mask = mask & ~missing_mask
            detailed[label] = data[clean_mask].to_dict(orient="records")
            combined_invalid_mask |= clean_mask

        # Collect other categories
        detailed["missing"] = data[missing_mask].to_dict(orient="records")
        if is_numeric_column(series):
            detailed["zero"] = data[zero_mask].to_dict(orient="records")

        # Now valid = not missing, not zero, not invalid
        valid_mask = ~(missing_mask | zero_mask | combined_invalid_mask) if is_numeric_column(series) else ~(missing_mask | combined_invalid_mask)
        detailed["valid"] = data[valid_mask].to_dict(orient="records")

        return detailed


    if groupBy:
        result["grouped"] = True
        result["analysis"] = indicatorFillRateGrouped(
            df, colName, groupBy, invalid_conditions, include_zero_as_separate_category
        )
        result["detailed_data"] = {
            group: get_detailed_data(group_df,colName,invalid_conditions)
            for group, group_df in df.groupby(groupBy)
        }
    else:
        result["grouped"] = False
        result["analysis"] = indicatorFillRate(
            df, colName, invalid_conditions, include_zero_as_separate_category
        )
        result["detailed_data"] = get_detailed_data(df,colName,invalid_conditions)

    return result


def frequencyTable(
    df: pd.DataFrame, 
    colName: str, 
    top_n: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a frequency table for a given column in a dataframe.

    Args:
    df (pd.DataFrame): Input dataframe
    colName (str): Name of the column to analyze
    top_n (Optional[int]): Sort order - "ascending", "descending", or None (default: None)

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: (Complete frequency table, Top n frequency table)
    """
    # Calculate value counts and reset index to make it a proper dataframe
    freqTable = df[colName].value_counts().reset_index()
    freqTable.columns = ["value", "Frequency"]

    # Calculate share
    total = len(df)
    freqTable["share"] = (freqTable["Frequency"] / total * 100).round(2)

    # Normalize order input
    if top_n:
        top_n = top_n.lower()

    # Sort by count descending (should already be sorted, but this ensures it)
    #freqTable = freqTable.sort_values("count", ascending=False).reset_index(drop=True)
    #topNFreq = freqTable if top_n is None or top_n == 0 else freqTable.head(top_n)
    
    # Order based on user input
    if top_n == "ascending":
        topNFreq = freqTable.sort_values("Frequency", ascending=True)
    elif top_n == "descending":
        topNFreq = freqTable.sort_values("Frequency", ascending=False)
    else: 
        topNFreq = freqTable.copy()  # No sorting

    return freqTable, topNFreq


def analyze_frequency_table(
    df: pd.DataFrame,
    colName: str,
    top_n: Optional[str] = None,
    groupBy: Optional[str] = None,
    filterBy: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Analyze frequency table in a dataframe column with optional grouping and filtering.

    Args:
    df (pd.DataFrame): Input dataframe
    colName (str): Name of the column to analyze
    top_n (Optional[str]): Order the  data to return separately (default: None)
    groupBy (Optional[str]): Name of the categorical column to group by (default: None)
    filterBy (Optional[Dict[str, str]]): Dictionary with column name as key and value to filter on (default: None)

    Returns:
    Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]]: Dictionary with analysis results
    """
    result = {}

    if filterBy:
        for col, value in filterBy.items():
            df = df[df[col] == value]
        result["filtered"] = True
    else:
        result["filtered"] = False

    if groupBy:

        # Normalize top_n input for consistency
        order = top_n.lower() if top_n else None

        # Create a frequency table for all combinations when groupBy is activated
        result["grouped"] = True
        grouped_freq_table = (
            df.groupby([groupBy, colName]).size().reset_index(name="count")
        )
        grouped_freq_table["share %"] = (
            grouped_freq_table["count"] / grouped_freq_table["count"].sum() * 100
        ).round(2)

        # Apply sorting based on order
        if order == "ascending":
            grouped_freq_table = grouped_freq_table.sort_values("count", ascending=True)
        elif order == "descending":
            grouped_freq_table = grouped_freq_table.sort_values("count", ascending=False)
        
        result["analysis"] = grouped_freq_table.reset_index(drop=True)
        top_n_freq = grouped_freq_table.copy()

        result["analysis"] = (grouped_freq_table, top_n_freq)

    else:
        result["grouped"] = False
        result["analysis"] = frequencyTable(df, colName, top_n)

    return result
