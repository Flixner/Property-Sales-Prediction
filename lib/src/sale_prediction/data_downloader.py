import pandas as pd
import requests


def get_query_by_table_id(table_id: str) -> str | None:
    """
    Retrieves the SQL query for a given table ID.

    This function uses a hardcoded mapping between table IDs and their respective SQL queries
    to prevent SQL injection risks. Each query corresponds to a predefined table ID, ensuring
    that only safe, validated queries are executed.

    Args:
        table_id (str): The unique identifier for the table.

    Returns:
        str | None: The corresponding SQL query for the table ID, or None if the table ID is not recognized.
    """
    return {
        "01651dab-2be7-40c6-a9d6-31254fe02e29": 'SELECT * FROM "01651dab-2be7-40c6-a9d6-31254fe02e29"',  # 2024
        "26d0ca6e-0a70-41fc-8e60-8c1a32343877": 'SELECT * FROM "26d0ca6e-0a70-41fc-8e60-8c1a32343877"',  # 2023
        "060179e8-0d9d-4071-aa54-fd62223842b6": 'SELECT * FROM "060179e8-0d9d-4071-aa54-fd62223842b6"',  # 2022
        "31f81cfe-b34f-496b-9361-2025514920cb": 'SELECT * FROM "31f81cfe-b34f-496b-9361-2025514920cb"',  # 2021
        "5ad3b44d-ba65-47eb-bd08-3f6cd07bf597": 'SELECT * FROM "5ad3b44d-ba65-47eb-bd08-3f6cd07bf597"',  # 2020
        "7c2f3357-8380-4cd7-9b50-67b0e554ff7d": 'SELECT * FROM "7c2f3357-8380-4cd7-9b50-67b0e554ff7d"',  # 2019
        "f083631f-e34e-4ad6-aba1-d6d7dd265170": 'SELECT * FROM "f083631f-e34e-4ad6-aba1-d6d7dd265170"',  # 2002-2018
    }.get(table_id)


def fetch_table_data(api_url: str, table_id: str, limit: int | None = None) -> pd.DataFrame:
    """
    Fetches data for a given table from the API.

    This function retrieves data for a specific table ID by executing a predefined SQL query.
    Queries are hardcoded to mitigate SQL injection risks. If a `limit` is provided, it appends
    a `LIMIT` clause to the query.

    Args:
        api_url (str): The base URL of the API.
        table_id (str): The unique identifier for the table.
        limit (int | None): The maximum number of rows to fetch. If None, fetches all rows.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the table data.

    Raises:
        ValueError: If the table ID is not recognized or the API response format is unexpected.
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    query = get_query_by_table_id(table_id=table_id)
    if query is None:
        raise ValueError(f"Invalid table ID: {table_id}")

    if limit:
        query += f" LIMIT {limit}"

    # Make the API request
    response = requests.get(api_url, params={"sql": query}, timeout=60)
    response.raise_for_status()

    # Parse the response data
    data = response.json()
    if "result" not in data or "records" not in data["result"]:
        raise ValueError(f"Unexpected response format for table {table_id}")

    # Convert records to DataFrame
    df = pd.DataFrame(data["result"]["records"])
    df = df.drop(columns=["_full_text"], errors="ignore")
    return df


def save_combined_data_to_csv(output_file: str, combined_df: pd.DataFrame) -> None:
    """Save the combined DataFrame to a CSV file.

    Args:
        output_file (str): Path to save the CSV file.
        combined_df (pd.DataFrame): DataFrame containing all combined table data.
    """
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")


def download_and_save_combined_tables(
    api_url: str, table_ids: list[str], output_file: str, limit: int | None = None, col_map: dict[str, str] | None = None
) -> None:
    """Download specified tables and save them as a single CSV file.

    Args:
        api_url (str): Base URL for the API.
        table_ids (List[str]): List of table IDs to download.
        output_file (str): Path to save the combined CSV file.
        limit (int, optional): Maximum number of rows to fetch. Defaults to None.
        col_map (dict[str, str], optional): Column mapping for renaming. Defaults to None.
    """
    dfs = []
    col_map = col_map if col_map else {}

    for table_id in table_ids:
        try:
            print(f"Fetching data for table: {table_id}")
            dfs.append(fetch_table_data(api_url, table_id, limit).rename(columns=col_map))
        except Exception as e:
            print(f"Error processing table {table_id}: {e}")

    save_combined_data_to_csv(output_file, pd.concat(dfs, ignore_index=True))


if __name__ == "__main__":
    API_URL = "https://data.milwaukee.gov/api/3/action/datastore_search_sql"
    OUTPUT_FILE = "data/full_dataset_2002_2024.csv"
    TABLE_IDS = [
        "01651dab-2be7-40c6-a9d6-31254fe02e29",  # 2024
        "26d0ca6e-0a70-41fc-8e60-8c1a32343877",  # 2023
        "060179e8-0d9d-4071-aa54-fd62223842b6",  # 2022
        "31f81cfe-b34f-496b-9361-2025514920cb",  # 2021
        "5ad3b44d-ba65-47eb-bd08-3f6cd07bf597",  # 2020
        "7c2f3357-8380-4cd7-9b50-67b0e554ff7d",  # 2019
        "f083631f-e34e-4ad6-aba1-d6d7dd265170",  # 2002 - 2018
    ]
    ROW_LIMIT = None
    COLUMN_MAP = {"Taxkey": "taxkey", "Nbhd": "nbhd", "Nr_of_rms": "Rooms", "Fin_sqft": "FinishedSqft"}

    download_and_save_combined_tables(API_URL, TABLE_IDS, OUTPUT_FILE, limit=ROW_LIMIT, col_map=COLUMN_MAP)
