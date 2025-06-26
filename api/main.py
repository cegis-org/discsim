import json
import pandas as pd
import numpy as np
import io
from typing import List
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    Query,
    Depends,
)
from fastapi.responses import Response, JSONResponse
from sqlalchemy.orm import Session
from api.models import (
    DropExportDuplicatesInput,
    DropExportDuplicatesResponse,
    ErrorHandlingInput,
    PreliminaryTestResponse,
    UniqueIDResponse,
    UniqueIDCheckInput,
    UniqueIDCheckResponse,
    L1SampleSizeInput,
    L2SampleSizeInput,
    ThirdPartySamplingInput,
)
from api.utils.administrative_data_quality_checklist import (
    analyze_frequency_table,
    analyze_indicator_fill_rate,
    analyze_missing_entries,
    analyze_zero_entries,
    findUniqueIDs,
    uniqueIDcheck,
    dropExportDuplicates,
    run_preliminary_tests,
)
from api.utils.pre_survey_analysis import (
    error_handling,
    l1_sample_size_calculator,
    l2_sample_size_calculator,
    third_party_sampling_strategy,
)
from api.utils.post_survey_analysis import calculate_discrepancy_scores
from api.utils.pseudo_code import anganwadi_center_data_anaylsis
from api.database import get_db, UploadedFile
from api.database import engine, Base
import chardet
import logging
import csv
from io import StringIO

app = FastAPI()
logger = logging.getLogger(__name__)

# Global variable to store the last processed data
last_processed_data = {"unique_rows": None, "duplicate_rows": None}

# Global variable to store the last deduplicated data
last_deduplicated_data = None


@app.on_event("startup")
async def startup_event():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Error creating tables: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}

def detect_delimiter(sample_text: str) -> str:
    """Detect whether comma or semicolon is the most likely delimiter."""
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text)
        return dialect.delimiter
    except csv.Error:
        # Fallback
        return ',' if sample_text.count(',') >= sample_text.count(';') else ';'
@app.post("/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form(...),
    db: Session = Depends(get_db)
):
    contents = await file.read()
    detection = chardet.detect(contents)
    encoding = detection["encoding"]
    encoding = encoding.lower() if encoding else None
    logger.info(f"Detected file encoding: '{encoding}'")
    try:
        if encoding not in ['utf-8', 'utf-8-sig', 'iso-8859-1', 'windows-1252', 'ascii']:
            logger.warning(f"Unsupported file encoding: {file.filename} (Detected: {encoding})")
            return JSONResponse(
                status_code=400, 
                content={"message": f"Unsupported file encoding: {encoding}"},
            )
        
        # Decode contents
        text_data = contents.decode(encoding)
        df_raw = pd.read_csv(io.StringIO(text_data), header=0)

        # If only one column exists, try splitting it
        if df_raw.shape[1] == 1:
            logger.warning(f"Only one column detected in {file.filename}. Attempting to split.")

            sample = text_data[:1000]  # Use first 1000 characters for detection
            detected_delim = detect_delimiter(sample)
            logger.warning(f"Detected delimiter: '{detected_delim}'")
            # Split the single column into columns
            # split_df = df_raw[0].str.split(detected_delim, expand=True)
            # Use first row as header -> include explicitly headers
            # split_df.columns = split_df.iloc[0].astype(str)
            # Drop the header row from data
            # df = split_df.iloc[1:].reset_index(drop=True)
            df = pd.read_csv(io.StringIO(text_data), delimiter=detected_delim, header=0)

        else:
            df = df_raw

        # Convert DataFrame back to CSV
        processed_csv = df.to_csv(index=False)
        

        # Check if a file with the same name and category already exists
        existing_file = (
            db.query(UploadedFile)
            .filter(UploadedFile.filename == file.filename, UploadedFile.category == category)
            .first()
        )

        if existing_file:
            return JSONResponse(
                status_code=409,  # Conflict
                content={
                    "message": f"'{file.filename}' already exists in category '{category}'.",
                    "id": existing_file.id,
                },
            )

        db_file = UploadedFile(filename=file.filename, content=processed_csv.encode(encoding), category=category)
        db.add(db_file)
        db.commit()
        db.refresh(db_file)

    except Exception as e:
        db.rollback()
        logger.error(f"Error saving file: {file.filename}. Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "details": "Error saving file in DB"},
        )

    logger.info(f"File uploaded successfully: {file.filename}")
    return {"message": "File uploaded successfully", "id": db_file.id}


@app.get("/list_files")
async def list_files(
    category: str = Query(...),
    db: Session = Depends(get_db)
):
    files = (
        db.query(UploadedFile.id, UploadedFile.filename, UploadedFile.upload_datetime)
        .filter(UploadedFile.category == category)
        .all()
    )
    return [
        {
            "id": file.id,
            "filename": file.filename,
            "upload_datetime": file.upload_datetime.isoformat(),
        }
        for file in files
    ]


@app.get("/get_file/{file_id}")
async def get_file(file_id: int, db: Session = Depends(get_db)):
    file = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file.content:
        raise HTTPException(status_code=400, detail="File content is empty or missing")

    detected = chardet.detect(file.content)
    encoding = detected["encoding"]
    # if not encoding:
    #     raise HTTPException(status_code=400, detail="Unable to detect file encoding")
    
    normalized_encoding = encoding.lower() if encoding else None
    accepted_encodings = ['utf-8', 'utf-8-sig', 'ascii']
    if normalized_encoding not in accepted_encodings:
        raise HTTPException(
            status_code=400,
            detail=f"File is not UTF-8 encoded. Encoding found: {encoding}"
        )
    
    try:
        decoded_content = file.content.decode(normalized_encoding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decoding file: {str(e)}")

    return {
        "filename": file.filename,
        "datetime": file.upload_datetime.isoformat(),
        "content": decoded_content
    }


@app.post("/preliminary_tests", response_model=PreliminaryTestResponse)
async def preliminary_tests(
    file: UploadFile = File(...),
    file_id: int = None,
    db: Session = Depends(get_db)
):
    try:
        if file:
            contents = await file.read()
        elif file_id:
            stored_file = (
                db.query(UploadedFile)
                .filter(UploadedFile.id == file_id)
                .first()
            )
            if not stored_file:
                raise HTTPException(status_code=404, detail="File not found")
            contents = stored_file.content
        else:
            raise HTTPException(
                status_code=400,
                detail="Either file or file_id must be provided"
            )

        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        result = run_preliminary_tests(df)
        return PreliminaryTestResponse(**result)
    except Exception as e:
        print(f"Error in preliminary_tests: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/find_unique_ids", response_model=List[UniqueIDResponse])
async def find_unique_ids(
    file: UploadFile = File(...),
    file_id: int = None,
    db: Session = Depends(get_db)
):
    try:
        if file:
            contents = await file.read()
        elif file_id:
            stored_file = (
                db.query(UploadedFile)
                .filter(UploadedFile.id == file_id)
                .first()
            )
            if not stored_file:
                raise HTTPException(status_code=404, detail="File not found")
            contents = stored_file.content
        else:
            raise HTTPException(
                status_code=400,
                detail="Either file or file_id must be provided"
            )

        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        result = findUniqueIDs(df.to_dict("records"))
        return [
            UniqueIDResponse(
                UniqueID=item["UniqueID"],
                Numeric_DataTypes=item["Numeric_DataTypes"]
            )
            for item in result
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/unique_id_check", response_model=UniqueIDCheckResponse)
async def unique_id_check(input_data: UniqueIDCheckInput):
    try:
        df = pd.DataFrame(input_data.data)
        result = uniqueIDcheck(df.to_dict("records"), input_data.columns)
        return UniqueIDCheckResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/drop_export_duplicates",
    response_model=DropExportDuplicatesResponse
)
async def drop_export_duplicates(
    file: UploadFile = File(...),
    file_id: int = None,
    input_data: str = Form(...),
    db: Session = Depends(get_db)
):
    global last_processed_data
    try:
        input_params = json.loads(input_data)
        input_model = DropExportDuplicatesInput(**input_params)

        if file:
            contents = await file.read()
        elif file_id:
            stored_file = (
                db.query(UploadedFile)
                .filter(UploadedFile.id == file_id)
                .first()
            )
            if not stored_file:
                raise HTTPException(status_code=404, detail="File not found")
            contents = stored_file.content
        else:
            raise HTTPException(
                status_code=400,
                detail="Either file or file_id must be provided"
            )

        df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")),
            keep_default_na=False, na_values=[""]
        )

        unique_rows, duplicate_rows = dropExportDuplicates(
            df,
            input_model.uidCol,
            input_model.keptRow,
            input_model.export,
            input_model.chunksize,
        )

        unique_count = len(unique_rows)
        duplicate_count = len(duplicate_rows) if duplicate_rows is not None else 0
        total_count = unique_count + duplicate_count
        percent_duplicates = (
            (duplicate_count / total_count) * 100 if total_count > 0 else 0
        )

        # Store the processed data
        last_processed_data["unique_rows"] = pd.DataFrame(unique_rows)
        last_processed_data["duplicate_rows"] = (
            pd.DataFrame(duplicate_rows) if duplicate_rows is not None else None
        )

        return DropExportDuplicatesResponse(
            unique_count=unique_count,
            duplicate_count=duplicate_count,
            percent_duplicates=percent_duplicates,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_processed_data")
async def get_processed_data(data_type: str = Query(...), filename: str = Query(...)):
    global last_processed_data
    if data_type not in ["unique", "duplicate"]:
        raise HTTPException(status_code=400, detail="Invalid data type")

    data = (
        last_processed_data["unique_rows"]
        if data_type == "unique"
        else last_processed_data["duplicate_rows"]
    )

    if data is not None:
        csv_data = data.to_csv(index=False)
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    else:
        raise HTTPException(status_code=404, detail=f"No {data_type} data available")


@app.get("/get_dataframe")
async def get_dataframe(data_type: str = Query(...)):
    global last_processed_data
    if data_type not in ["unique", "duplicate"]:
        raise HTTPException(status_code=400, detail="Invalid data type")

    data = (
        last_processed_data["unique_rows"]
        if data_type == "unique"
        else last_processed_data["duplicate_rows"]
    )

    if data is not None:
        return data.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(
            orient="records"
        )
    else:
        raise HTTPException(status_code=404, detail=f"No {data_type} data available")


@app.post("/drop_export_duplicate_rows", response_model=DropExportDuplicatesResponse)
async def drop_export_duplicate_rows(
    file: UploadFile = File(...), 
    # input_data: str = Form(...)
):
    global last_processed_data
    try:
        # input_params = json.loads(input_data)
        # kept_row = input_params.get("keptRow", "first")
        # export = input_params.get("export", True)

        contents = await file.read()
        df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")), keep_default_na=False, na_values=[""]
        )

        # Process duplicates
        #if kept_row == "none":
        unique_rows = df.drop_duplicates(keep=False)
        duplicate_rows = df[df.duplicated(keep=False)]
        # else:
        #     unique_rows = df.drop_duplicates(keep=kept_row)
        #     duplicate_rows = df[df.duplicated(keep=False)] if export else None

        unique_count = len(unique_rows)
        duplicate_count = len(duplicate_rows) if duplicate_rows is not None else 0
        total_count = unique_count + duplicate_count
        percent_duplicates = (
            (duplicate_count / total_count) * 100 if total_count > 0 else 0
        )

        # Store the processed data
        last_processed_data["unique_rows"] = unique_rows
        last_processed_data["duplicate_rows"] = duplicate_rows

        return DropExportDuplicatesResponse(
            unique_count=unique_count,
            duplicate_count=duplicate_count,
            percent_duplicates=percent_duplicates,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/missing_entries")
async def missing_entries(
    file: UploadFile = File(...),
    file_id: int = None,
    input_data: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        if file:
            contents = await file.read()
        elif file_id:
            stored_file = (
                db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
            )
            if not stored_file:
                raise HTTPException(status_code=404, detail="File not found")
            contents = stored_file.content
        else:
            raise HTTPException(
                status_code=400, detail="Either file or file_id must be provided"
            )

        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), index_col=False)

        # Parse the input data
        input_data = json.loads(input_data)
        column_to_analyze = input_data["column_to_analyze"]
        group_by = input_data.get("group_by")
        filter_by = input_data.get("filter_by")

        # Validate input
        if column_to_analyze not in df.columns:
            raise ValueError(f"Column '{column_to_analyze}' not found in the dataset")

        if group_by and group_by not in df.columns:
            raise ValueError(f"Group by column '{group_by}' not found in the dataset")

        if filter_by:
            for col, value in filter_by.items():
                if col not in df.columns:
                    raise ValueError(f"Filter column '{col}' not found in the dataset")
                if df[df[col] == value].empty:
                    raise ValueError(f"No data found for filter: {col} = {value}")

        # Perform the analysis
        result = analyze_missing_entries(df, column_to_analyze, group_by, filter_by)
        # Convert numpy types to Python native types for JSON serialization
        def safe_convert(entry):
            count = int(entry[0])
            percent = (
                float(entry[1]) if entry[1] is not None and not np.isnan(entry[1]) and not np.isinf(entry[1]) else None
            )
            total = int(entry[2]) if entry[2] is not None else 0
            return (count, percent, total)
        if isinstance(result["analysis"], dict):
            result["analysis"] = {k: safe_convert(v) for k, v in result["analysis"].items()}
            result["total_rows"] = df.shape[0]
            result["zero_entries"] = int((df[group_by] == 0).sum())
        else:
            result["analysis"] = safe_convert(result["analysis"])

        # Include rows with missing entries
        missing_rows = df[df[column_to_analyze].isna()]
        missing_rows_table = missing_rows.reset_index().to_dict(orient="records")

        # Convert any non-serializable values to None
        for row in missing_rows_table:
            for key, value in row.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    row[key] = None
                elif pd.isna(value):
                    row[key] = None

        result["missing_entries_table"] = missing_rows_table

        return JSONResponse(content=result)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/zero_entries")
async def zero_entries(
    file: UploadFile = File(...),
    file_id: int = None,
    input_data: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        if file:
            contents = await file.read()
        elif file_id:
            stored_file = (
                db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
            )
            if not stored_file:
                raise HTTPException(status_code=404, detail="File not found")
            contents = stored_file.content
        else:
            raise HTTPException(
                status_code=400, detail="Either file or file_id must be provided"
            )

        # Load the CSV file into a DataFrame
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Parse the input data
        input_data = json.loads(input_data)
        column_to_analyze = input_data["column_to_analyze"]
        group_by = input_data.get("group_by")
        filter_by = input_data.get("filter_by")

        # Validate input
        if column_to_analyze not in df.columns:
            raise ValueError(
                f"Column '{column_to_analyze}' not found in the dataset: {df.columns.tolist()}"
            )

        if group_by and group_by not in df.columns:
            raise ValueError(f"Group by column '{group_by}' not found in the dataset")

        if filter_by:
            for col in filter_by.keys():
                if col not in df.columns:
                    raise ValueError(f"Filter column '{col}' not found in the dataset")

        # Perform the analysis
        result = analyze_zero_entries(df, column_to_analyze, group_by, filter_by)

        # Convert numpy types to Python native types for JSON serialization
        if isinstance(result["analysis"], dict):
            result["analysis"] = {
                k: (int(v[0]), float(v[1]), int(v[2])) for k, v in result["analysis"].items()
            }
            result["total_rows"] = df.shape[0]
            result["zero_entries"] = int((df[column_to_analyze] == 0).sum())
        else:
            result["analysis"] = (
                int(result["analysis"][0]),
                float(result["analysis"][1]),
                int(result["analysis"][2]),
            )

        # Convert any non-serializable values to None in zero_entries_table
        for row in result["zero_entries_table"]:
            for key, value in row.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    row[key] = None
                elif pd.isna(value):
                    row[key] = None

        return JSONResponse(content=result)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/indicator_fill_rate")
async def indicator_fill_rate(
    file: UploadFile = File(...),
    file_id: int = None,
    input_data: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        if file:
            contents = await file.read()
        elif file_id:
            stored_file = (
                db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
            )
            if not stored_file:
                raise HTTPException(status_code=404, detail="File not found")
            contents = stored_file.content
        else:
            raise HTTPException(
                status_code=400, detail="Either file or file_id must be provided"
            )

        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Parse the input data
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in input_data")

        # Extract required field
        column_to_analyze = input_data.get("column_to_analyze")
        if not column_to_analyze:
            raise HTTPException(status_code=422, detail="'column_to_analyze' is required")
        
        group_by = input_data.get("group_by")
        filter_by = input_data.get("filter_by")
        invalid_conditions = input_data.get("invalid_conditions", [])
        include_zero_as_separate_category = input_data.get(
            "include_zero_as_separate_category", True
        )
        # Validate input
        if column_to_analyze not in df.columns:
            raise ValueError(f"Column '{column_to_analyze}' not found in the dataset")

        if group_by and group_by not in df.columns:
            raise ValueError(f"Group by column '{group_by}' not found in the dataset")

        if filter_by:
            for col in filter_by.keys():
                if col not in df.columns:
                    raise ValueError(f"Filter column '{col}' not found in the dataset")

        # Perform the analysis
        result = analyze_indicator_fill_rate(
            df,
            column_to_analyze,
            group_by,
            filter_by,
            invalid_conditions,
            include_zero_as_separate_category,
        )

        # Convert DataFrame to dict for JSON serialization
        if isinstance(result["analysis"], dict):
            result["analysis"] = {k: v.to_dict() for k, v in result["analysis"].items()}
        elif isinstance(result["analysis"], pd.DataFrame):
            if include_zero_as_separate_category:
                result["analysis"] = result["analysis"].to_dict()
            else:
                result["analysis"] = result["analysis"][
                    result["analysis"]["Category"] != "Zero"
                ].to_dict()
        else:
            # If it's neither a dict nor a DataFrame, convert to a serializable format
            result["analysis"] = json.loads(json.dumps(result["analysis"], default=str))

        # Convert any non-serializable values to None in detailed_data
        def clean_data(data):
            if isinstance(data, dict):
                return {k: clean_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_data(item) for item in data]
            elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
                return None
            elif pd.isna(data):
                return None
            else:
                return data

        result["detailed_data"] = clean_data(result["detailed_data"])

        return JSONResponse(content=result)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/frequency_table")
async def frequency_table(
    file: UploadFile = File(...),
    file_id: int = None,
    input_data: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        if file:
            contents = await file.read()
        elif file_id:
            stored_file = (
                db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
            )
            if not stored_file:
                raise HTTPException(status_code=404, detail="File not found")
            contents = stored_file.content
        else:
            raise HTTPException(
                status_code=400, detail="Either file or file_id must be provided"
            )

        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Parse the input data
        input_data = json.loads(input_data)
        column_to_analyze = input_data["column_to_analyze"]
        top_n = input_data.get("top_n", None)
        group_by = input_data.get("group_by")
        filter_by = input_data.get("filter_by")

        # Validate input
        if column_to_analyze not in df.columns:
            raise ValueError(f"Column '{column_to_analyze}' not found in the dataset")

        if group_by and group_by not in df.columns:
            raise ValueError(f"Group by column '{group_by}' not found in the dataset")

        if filter_by:
            for col in filter_by.keys():
                if col not in df.columns:
                    raise ValueError(f"Filter column '{col}' not found in the dataset")

        # Perform the analysis
        result = analyze_frequency_table(
            df, column_to_analyze, top_n, group_by, filter_by
        )

        #send number of rows analysed
        result["total"] = len(df)

        # Convert DataFrame to dict for JSON serialization
        if isinstance(result["analysis"], dict):
            result["analysis"] = {
                k: (v[0].to_dict(orient="records"), v[1].to_dict(orient="records"))
                for k, v in result["analysis"].items()
            }
        else:
            result["analysis"] = (
                result["analysis"][0].to_dict(orient="records"),
                result["analysis"][1].to_dict(orient="records"),
            )

        return JSONResponse(content=result)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/error-handling")
async def check_errors(input_data: ErrorHandlingInput):
    error_status, error_message = error_handling(input_data.params)
    return {"status": error_status, "message": error_message}


@app.post("/l1-sample-size")
async def calculate_l1_sample_size(input_data: L1SampleSizeInput):
    result = l1_sample_size_calculator(input_data.dict())
    if result["status"] == 0:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/l2-sample-size")
async def calculate_l2_sample_size(input_data: L2SampleSizeInput):
    result = l2_sample_size_calculator(input_data.dict())
    if result["status"] == 0:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/third-party-sampling")
async def predict_third_party_sampling(input_data: ThirdPartySamplingInput):
    result = third_party_sampling_strategy(input_data.dict())
    if result["status"] == 0:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/post_survey_analysis")
async def post_survey_analysis(
    file: UploadFile = File(...),
    margin_of_error_height: float = Form(0.0),
    margin_of_error_weight: float = Form(0.0)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Perform discrepancy calculations
        result = calculate_discrepancy_scores(df, margin_of_error_height, margin_of_error_weight)

        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error in post_survey_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/pseudo_code")
async def pseudo_code(
    file: UploadFile = File(...)
):
    try:
        if file.content_type != "text/csv":
            raise HTTPException(status_code=400, detail="Please upload a CSV file.")
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        result = anganwadi_center_data_anaylsis(df)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error in pseudo_code_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))