# DiscSim API Documentation

Welcome to the DiscSim API documentation. This API provides a suite of tools for data analysis, deduplication, and sampling strategies. Below you'll find detailed information about each endpoint, its purpose, and how to use it.

## Table of Contents

1. [Data Analysis](#data-analysis)
2. [Deduplication](#deduplication)
3. [Sampling Strategies](#sampling-strategies)

## Data Analysis

### Preliminary Tests

Analyze your dataset for basic integrity and structure.

**Endpoint:** `POST /preliminary_tests`

**Request:**
- File upload: CSV file of the dataset

**Response:**
```json
{
  "status": 0,
  "error_code": null,
  "message": "Success",
  "warnings": ["Column 'age' has missing values"]
}
```
**Dashboard Working:**

![image](https://github.com/user-attachments/assets/b04ee57a-652a-41ac-855b-3eb9352e0901)

### Find Unique IDs

Identify potential unique identifiers in your dataset.

**Endpoint:** `POST /find_unique_ids`

**Request:**
- File upload: CSV file of the dataset

**Response:**
```json
[
  {
    "UniqueID": "customer_id",
    "Numeric_DataTypes": true
  },
  {
    "UniqueID": "email",
    "Numeric_DataTypes": false
  }
]
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/0284f6aa-40e0-4daa-80c0-123362249a7a)

### Unique ID Check

Verify if specified columns can serve as unique identifiers.

**Endpoint:** `POST /unique_id_check`

**Request:**
```json
{
  "data": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}],
  "columns": ["id"]
}
```

**Response:**
```json
{
  "result": true
}
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/56b3d086-f7ee-4525-a837-08d56b09ec85)

### Missing Entries Analysis

Analyze missing data in a specific column.

**Endpoint:** `POST /missing_entries`

**Request:**
- File upload: CSV file of the dataset
- Form data: JSON string with analysis parameters
  ```json
  {
    "column_to_analyze": "age",
    "group_by": "gender",
    "filter_by": {"country": "USA"}
  }
  ```

**Response:**
```json
{
  "filtered": true,
  "grouped": true,
  "analysis": {
    "Male": [100, 5.2],
    "Female": [80, 4.1]
  },
  "missing_entries_table": [
    {"id": 1, "age": null, "gender": "Male"}
  ]
}
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/dbbe4f41-9a86-49f0-b4fe-b458ab7defc6)

### Zero Entries Analysis

Analyze zero values in a specific column.

**Endpoint:** `POST /zero_entries`

**Request:**
- File upload: CSV file of the dataset
- Form data: JSON string with analysis parameters (similar to missing entries)

**Response:** Similar to missing entries analysis

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/61161de5-008e-431b-8c76-620afa57958a)

### Indicator Fill Rate Analysis

Analyze the fill rate of indicators in your dataset.

**Endpoint:** `POST /indicator_fill_rate`

**Request:**
- File upload: CSV file of the dataset
- Form data: JSON string with analysis parameters
  ```json
  {
    "column_to_analyze": "income",
    "group_by": "job_category",
    "filter_by": {"age": "> 18"},
    "invalid_condition": "> 1000000",
    "include_zero_as_separate_category": true
  }
  ```

**Response:**
```json
{
  "filtered": true,
  "grouped": true,
  "analysis": {
    "Manager": {
      "Missing": [10, 2.5],
      "Zero": [5, 1.25],
      "Invalid": [2, 0.5],
      "Valid": [383, 95.75]
    }
  },
  "detailed_data": {
    "Manager": {
      "missing": [{"id": 1, "income": null}],
      "zero": [{"id": 2, "income": 0}],
      "invalid": [{"id": 3, "income": 2000000}],
      "valid": [{"id": 4, "income": 50000}]
    }
  }
}
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/a3ed95ad-d4ca-4b29-9697-f77ac9490730)

### Frequency Table Analysis

Generate a frequency table for a specific column.

**Endpoint:** `POST /frequency_table`

**Request:**
- File upload: CSV file of the dataset
- Form data: JSON string with analysis parameters
  ```json
  {
    "column_to_analyze": "product_category",
    "top_n": 5,
    "group_by": "store_location",
    "filter_by": {"year": 2023}
  }
  ```

**Response:**
```json
{
  "filtered": true,
  "grouped": true,
  "analysis": [
    {
      "store_location": "New York",
      "product_category": "Electronics",
      "count": 1500,
      "share %": 30.5
    }
  ]
}
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/b2138fda-fa98-4cc6-a87d-abb4a7eb8a74)

## Deduplication

### Drop and Export Duplicates

Remove duplicates from your dataset based on specified criteria.

**Endpoint:** `POST /drop_export_duplicates`

**Request:**
- File upload: CSV file of the dataset
- Form data: JSON string with deduplication parameters
  ```json
  {
    "uidCol": "customer_id",
    "keptRow": "first",
    "export": true,
    "chunksize": 10000
  }
  ```

**Response:**
```json
{
  "unique_count": 9500,
  "duplicate_count": 500,
  "percent_duplicates": 5.0
}
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/8552e78a-3163-4388-97a2-2a78640c017c)

### Get Processed Data

Retrieve the results of the deduplication process.

**Endpoint:** `GET /get_processed_data`

**Query Parameters:**
- `data_type`: "unique" or "duplicate"
- `filename`: Desired filename for the download

**Response:** CSV file download

### Get DataFrame as JSON

Retrieve the processed data in JSON format.

**Endpoint:** `GET /get_dataframe`

**Query Parameters:**
- `data_type`: "unique" or "duplicate"

**Response:** JSON array of records

### Duplicate Analysis

Analyze the extent of duplication in your dataset.

**Endpoint:** `POST /duplicate_analysis`

**Request:**
- File upload: CSV file of the dataset

**Response:**
```json
{
  "num_duplicates": 500,
  "percent_duplicates": 5.0
}
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/41425f70-3699-4cb0-ac7a-15f9947c72d9)

### Remove Duplicates

Remove duplicates using a specified strategy.

**Endpoint:** `POST /remove_duplicates`

**Request:**
- File upload: CSV file of the dataset
- Form data: 
  - `remove_option`: "Keep first occurrence" or "Drop all occurrences"

**Response:**
```json
{
  "original_count": 10000,
  "deduplicated_count": 9500,
  "dropped_count": 500
}
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/f0bb82dd-34e2-4630-b2b4-8050fba88f5b)

### Get Deduplicated Data

Download the deduplicated dataset.

**Endpoint:** `GET /get_deduplicated_data`

**Query Parameters:**
- `filename`: Desired filename for the download

**Response:** CSV file download

## Sampling Strategies

### L1 Sample Size Calculator

Calculate the sample size for Level 1 analysis.

**Endpoint:** `POST /l1-sample-size`

**Request:**
```json
{
  "level_test": "Block",
  "n_subs_per_block": 100,
  "n_blocks_per_district": 10,
  "n_district": 5,
  "percent_punish": 10,
  "percent_guarantee": 5,
  "confidence": 0.95,
  "min_disc": 0,
  "max_disc": 1,
  "mean_disc": 0.5,
  "std_disc": 0.1,
  "distribution": "normal",
  "min_n_samples": 10,
  "max_n_samples": 1000,
  "n_simulations": 1000
}
```

**Response:**
```json
{
  "status": 1,
  "message": "L1 sample size calculated successfully.",
  "value": 250
}
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/7d0beb5c-e8c5-472e-9cdb-330e744ac246)

### L2 Sample Size Calculator

Calculate the sample size for Level 2 analysis.

**Endpoint:** `POST /l2-sample-size`

**Request:**
```json
{
  "level_test": "District",
  "n_subs_per_block": 100,
  "n_blocks_per_district": 10,
  "n_district": 5,
  "average_truth_score": 0.7,
  "variance_across_blocks": 0.1,
  "total_samples": 10000
}
```

**Response:**
```json
{
  "status": 1,
  "message": "L2 sample size calculated successfully.",
  "value": {
    "true_disc": [0.65, 0.72, 0.68],
    "meas_disc": [0.64, 0.73, 0.67],
    "n_samples": 200
  }
}
```

**Dashboard Working:**


### Third-Party Sampling Strategy

Generate a third-party sampling strategy recommendation.

**Endpoint:** `POST /third-party-sampling`

**Request:**
```json
{
  "level_test": "State",
  "n_subs_per_block": 100,
  "n_blocks_per_district": 10,
  "n_district": 5,
  "average_truth_score": 0.7,
  "variance_across_blocks": 0.1,
  "total_samples": 10000,
  "n_simulations": 1000,
  "percent_blocks_plot": 20
}
```

**Response:**
```json
{
  "status": 1,
  "message": "3P Sampling Strategy calculated successfully.",
  "value": {
    "true_disc": [0.65, 0.72, 0.68],
    "results": [
      {
        "n_sub_tested": 1,
        "n_samples": 200,
        "meas_disc": [[0.64, 0.73, 0.67]]
      }
    ],
    "n_blocks_plot": 10,
    "figure": "{...}" // JSON string of Plotly figure
  }
}
```

**Dashboard Working:**

![image](https://github.com/user-attachments/assets/e0481047-7edf-4434-ac40-c817319d79c5)
