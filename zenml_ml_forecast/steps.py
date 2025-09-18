import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing_extensions import Annotated
from zenml import step
from zenml.types import HTMLString

logger = logging.getLogger(__name__)


@step
def load_data() -> pd.DataFrame:
    """Load synthetic retail sales data for forecasting."""
    data_dir = os.path.join(os.getcwd(), "data")
    sales_path = os.path.join(data_dir, "sales.csv")

    if os.path.exists(sales_path):
        # Load real data if available
        sales_df = pd.read_csv(sales_path)
        logger.info(f"Loaded {len(sales_df)} sales records from file.")
    else:
        logger.info("Generating synthetic retail sales data...")
        # Create synthetic dataset with retail patterns
        np.random.seed(42)  # For reproducibility

        # Generate date range for 3 months
        date_range = pd.date_range("2024-01-01", periods=90, freq="D")

        # Create stores and items
        stores = ["Store_1", "Store_2"]
        items = ["Item_A", "Item_B", "Item_C"]

        records = []
        for date in date_range:
            # Calendar features
            is_weekend = 1 if date.dayofweek >= 5 else 0
            is_holiday = 1 if date.day == 1 or date.day == 15 else 0
            is_promo = 1 if 10 <= date.day <= 20 else 0

            for store in stores:
                for item in items:
                    # Base demand with factors
                    base_demand = 100
                    store_factor = 1.5 if store == "Store_1" else 0.8
                    item_factor = (
                        1.2
                        if item == "Item_A"
                        else 1.0
                        if item == "Item_B"
                        else 0.7
                    )
                    weekday_factor = 1.5 if is_weekend else 1.0
                    holiday_factor = 2.0 if is_holiday else 1.0
                    promo_factor = 1.8 if is_promo else 1.0

                    # Add random noise
                    noise = np.random.normal(1, 0.1)

                    # Calculate final sales
                    sales = int(
                        base_demand
                        * store_factor
                        * item_factor
                        * weekday_factor
                        * holiday_factor
                        * promo_factor
                        * noise
                    )
                    sales = max(0, sales)

                    records.append(
                        {
                            "date": date,
                            "store": store,
                            "item": item,
                            "sales": sales,
                        }
                    )

        # Create DataFrame
        sales_df = pd.DataFrame(records)

        # Save synthetic data
        os.makedirs(data_dir, exist_ok=True)
        sales_df.to_csv(sales_path, index=False)

    return sales_df


@step
def preprocess_data(
    sales_data: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[
    Annotated[Dict[str, pd.DataFrame], "training_data"],
    Annotated[Dict[str, pd.DataFrame], "testing_data"],
    Annotated[List[str], "series_identifiers"],
]:
    """Prepare data for forecasting with Prophet.

    Args:
        sales_data: Raw sales data with date, store, item, and sales columns
        test_size: Proportion of data to use for testing

    Returns:
        train_data_dict: Dictionary of training dataframes for each series
        test_data_dict: Dictionary of test dataframes for each series
        series_ids: List of unique series identifiers (store-item combinations)
    """
    logger.info(f"Preprocessing sales data with shape: {sales_data.shape}")

    # Convert date to datetime
    sales_data["date"] = pd.to_datetime(sales_data["date"])

    # Create unique series ID for each store-item combination
    sales_data["series_id"] = sales_data["store"] + "-" + sales_data["item"]

    # Get list of unique series
    series_ids = sales_data["series_id"].unique().tolist()
    logger.info(f"Found {len(series_ids)} unique store-item combinations")

    # Create Prophet-formatted dataframes (ds, y) for each series
    train_data_dict = {}
    test_data_dict = {}

    for series_id in series_ids:
        # Filter data for this series
        series_data = sales_data[sales_data["series_id"] == series_id].copy()

        # Sort by date and drop any duplicates
        series_data = series_data.sort_values("date").drop_duplicates(
            subset=["date"]
        )

        # Rename columns for Prophet
        prophet_data = series_data[["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        # Ensure no NaN values
        prophet_data = prophet_data.dropna()

        if len(prophet_data) < 2:
            logger.info(
                f"WARNING: Not enough data for series {series_id}, skipping"
            )
            continue

        # Make sure we have at least one point in test set
        min_test_size = max(1, int(len(prophet_data) * test_size))

        if len(prophet_data) <= min_test_size:
            # If we don't have enough data, use half for training and half for testing
            cutoff_idx = len(prophet_data) // 2
        else:
            cutoff_idx = len(prophet_data) - min_test_size

        # Split into train and test
        train_data = prophet_data.iloc[:cutoff_idx].copy()
        test_data = prophet_data.iloc[cutoff_idx:].copy()

        # Ensure we have data in both splits
        if len(train_data) == 0 or len(test_data) == 0:
            logger.info(
                f"WARNING: Empty split for series {series_id}, skipping"
            )
            continue

        # Store in dictionaries
        train_data_dict[series_id] = train_data
        test_data_dict[series_id] = test_data

        logger.info(
            f"Series {series_id}: {len(train_data)} train points, {len(test_data)} test points"
        )

    if not train_data_dict:
        raise ValueError("No valid series data after preprocessing!")

    # Get a sample series to print details
    sample_id = next(iter(train_data_dict))
    sample_train = train_data_dict[sample_id]
    sample_test = test_data_dict[sample_id]

    logger.info(f"Sample series {sample_id}:")
    logger.info(f"  Train data shape: {sample_train.shape}")
    logger.info(
        f"  Train date range: {sample_train['ds'].min()} to {sample_train['ds'].max()}"
    )
    logger.info(f"  Test data shape: {sample_test.shape}")
    logger.info(
        f"  Test date range: {sample_test['ds'].min()} to {sample_test['ds'].max()}"
    )

    return train_data_dict, test_data_dict, series_ids


@step
def validate_data(
    sales_data: pd.DataFrame, calendar_data: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "sales_data_validated"],
    Annotated[pd.DataFrame, "calendar_data_validated"],
]:
    """Validate retail sales data, checking for common issues like:
    - Missing values
    - Negative sales
    - Duplicate records
    - Date continuity
    - Extreme outliers
    """
    sales_df = sales_data
    calendar_df = calendar_data

    # Check for missing values in critical fields
    for df_name, df in [("Sales", sales_df), ("Calendar", calendar_df)]:
        if df.isnull().any().any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            logger.info(
                f"Warning: {df_name} data contains missing values in columns: {missing_cols}"
            )
            # Fill missing values appropriately based on column type
            for col in missing_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # For numeric columns, fill with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # For categorical/text columns, fill with most common value
                    df[col] = df[col].fillna(
                        df[col].mode()[0]
                        if not df[col].mode().empty
                        else "UNKNOWN"
                    )

    # Check for and fix negative sales (a common data quality issue in retail)
    neg_sales = (sales_df["sales"] < 0).sum()
    if neg_sales > 0:
        logger.info(
            f"Warning: Found {neg_sales} records with negative sales. Setting to zero."
        )
        sales_df.loc[sales_df["sales"] < 0, "sales"] = 0

    # Check for duplicate records
    duplicates = sales_df.duplicated(subset=["date", "store", "item"]).sum()
    if duplicates > 0:
        logger.info(
            f"Warning: Found {duplicates} duplicate store-item-date records. Keeping the last one."
        )
        sales_df = sales_df.drop_duplicates(
            subset=["date", "store", "item"], keep="last"
        )

    # Check for date continuity in calendar
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    date_diff = calendar_df["date"].diff().dropna()
    if not (date_diff == pd.Timedelta(days=1)).all():
        logger.info(
            "Warning: Calendar dates are not continuous. Some days may be missing."
        )

    # Detect extreme outliers (values > 3 std from mean within each item-store combination)
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    outlier_count = 0

    # Group by store and item to identify outliers within each time series
    for (store, item), group in sales_df.groupby(["store", "item"]):
        mean_sales = group["sales"].mean()
        std_sales = group["sales"].std()

        if std_sales > 0:  # Avoid division by zero
            # Calculate z-score
            z_scores = (group["sales"] - mean_sales) / std_sales

            # Flag extreme outliers (|z| > 3)
            outlier_mask = abs(z_scores) > 3
            outlier_count += outlier_mask.sum()

            # Cap outliers (winsorize) rather than removing them
            if outlier_mask.any():
                cap_upper = mean_sales + 3 * std_sales
                sales_df.loc[
                    group[outlier_mask & (group["sales"] > cap_upper)].index,
                    "sales",
                ] = cap_upper

    if outlier_count > 0:
        logger.info(
            f"Warning: Detected and capped {outlier_count} extreme sales outliers."
        )

    # Ensure all dates in sales exist in calendar
    if not set(sales_df["date"].dt.date).issubset(
        set(calendar_df["date"].dt.date)
    ):
        logger.info(
            "Warning: Some sales dates don't exist in the calendar data."
        )

    return sales_df, calendar_df



@step
def visualize_sales_data(
    sales_data: pd.DataFrame,
    train_data_dict: Dict[str, pd.DataFrame],
    test_data_dict: Dict[str, pd.DataFrame],
    series_ids: List[str],
) -> Annotated[HTMLString, "sales_visualization"]:
    """Create interactive visualizations of historical sales patterns.

    Args:
        sales_data: Raw sales data with date, store, item, and sales columns
        train_data_dict: Dictionary of training dataframes for each series
        test_data_dict: Dictionary of test dataframes for each series
        series_ids: List of unique series identifiers

    Returns:
        HTML visualization dashboard of historical sales patterns
    """
    # Ensure date column is in datetime format
    sales_data = sales_data.copy()
    sales_data["date"] = pd.to_datetime(sales_data["date"])

    # Create HTML with multiple visualizations
    html_parts = []
    html_parts.append("""
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f9f9f9;
            }
            .dashboard {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .section {
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            h1, h2, h3 {
                color: #333;
            }
            .insights {
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
            }
            .chart-container {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="section">
                <h1>Retail Sales Historical Data Analysis</h1>
                <p>Interactive visualization of sales patterns across stores and products.</p>
            </div>
    """)

    # Create overview metrics
    total_sales = sales_data["sales"].sum()
    avg_daily_sales = sales_data.groupby("date")["sales"].sum().mean()
    num_stores = sales_data["store"].nunique()
    num_items = sales_data["item"].nunique()
    min_date = sales_data["date"].min().strftime("%Y-%m-%d")
    max_date = sales_data["date"].max().strftime("%Y-%m-%d")
    date_range = f"{min_date} to {max_date}"

    html_parts.append(f"""
            <div class="section">
                <h2>Dataset Overview</h2>
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                    <div style="flex: 1; min-width: 200px; background: #f0f8ff; margin: 10px; padding: 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1);">
                        <h3>Total Sales</h3>
                        <p style="font-size: 24px; font-weight: bold;">{total_sales:,.0f} units</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: #fff8f0; margin: 10px; padding: 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1);">
                        <h3>Avg. Daily Sales</h3>
                        <p style="font-size: 24px; font-weight: bold;">{avg_daily_sales:,.1f} units</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: #f0fff8; margin: 10px; padding: 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1);">
                        <h3>Stores × Items</h3>
                        <p style="font-size: 24px; font-weight: bold;">{num_stores} × {num_items}</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: #f8f0ff; margin: 10px; padding: 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1);">
                        <h3>Date Range</h3>
                        <p style="font-size: 18px; font-weight: bold;">{date_range}</p>
                    </div>
                </div>
            </div>
    """)

    # 1. Time Series - Overall Sales Trend
    df_daily = sales_data.groupby("date")["sales"].sum().reset_index()
    fig_trend = px.line(
        df_daily,
        x="date",
        y="sales",
        title="Daily Total Sales Across All Stores and Products",
        template="plotly_white",
    )
    fig_trend.update_traces(line=dict(width=2))
    fig_trend.update_layout(
        xaxis_title="Date", yaxis_title="Total Sales (units)", height=500
    )
    trend_html = fig_trend.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Overall Sales Trend</h2>
                {trend_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Observe weekly patterns and special events that impact overall sales volume.</p>
                </div>
            </div>
    """)

    # 2. Store Comparison
    store_sales = (
        sales_data.groupby(["date", "store"])["sales"].sum().reset_index()
    )
    fig_stores = px.line(
        store_sales,
        x="date",
        y="sales",
        color="store",
        title="Sales Comparison by Store",
        template="plotly_white",
    )
    fig_stores.update_layout(
        xaxis_title="Date", yaxis_title="Total Sales (units)", height=500
    )
    stores_html = fig_stores.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Store Comparison</h2>
                {stores_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Compare performance across different stores to identify top performers and potential issues.</p>
                </div>
            </div>
    """)

    # 3. Product Performance
    item_sales = (
        sales_data.groupby(["date", "item"])["sales"].sum().reset_index()
    )
    fig_items = px.line(
        item_sales,
        x="date",
        y="sales",
        color="item",
        title="Sales Comparison by Product",
        template="plotly_white",
    )
    fig_items.update_layout(
        xaxis_title="Date", yaxis_title="Total Sales (units)", height=500
    )
    items_html = fig_items.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Product Performance</h2>
                {items_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Identify best-selling products and those with unique seasonal patterns.</p>
                </div>
            </div>
    """)

    # 4. Weekly Patterns
    sales_data["day_of_week"] = sales_data["date"].dt.day_name()
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekly_pattern = (
        sales_data.groupby("day_of_week")["sales"]
        .mean()
        .reindex(day_order)
        .reset_index()
    )

    fig_weekly = px.bar(
        weekly_pattern,
        x="day_of_week",
        y="sales",
        title="Average Sales by Day of Week",
        template="plotly_white",
        color="sales",
        color_continuous_scale="Blues",
    )
    fig_weekly.update_layout(
        xaxis_title="", yaxis_title="Average Sales (units)", height=500
    )
    weekly_html = fig_weekly.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Weekly Patterns</h2>
                {weekly_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Identify peak sales days to optimize inventory and staffing.</p>
                </div>
            </div>
    """)

    # 5. Sample Store-Item Combinations
    # Select 3 random series to display
    sample_series = np.random.choice(
        series_ids, size=min(3, len(series_ids)), replace=False
    )

    # Create subplots for train/test visualization
    fig_samples = make_subplots(
        rows=len(sample_series),
        cols=1,
        subplot_titles=[f"Series: {series_id}" for series_id in sample_series],
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    for i, series_id in enumerate(sample_series):
        train_data = train_data_dict[series_id]
        test_data = test_data_dict[series_id]

        # Add train data
        fig_samples.add_trace(
            go.Scatter(
                x=train_data["ds"],
                y=train_data["y"],
                mode="lines+markers",
                name=f"{series_id} (Training)",
                line=dict(color="blue"),
                legendgroup=series_id,
                showlegend=(i == 0),
            ),
            row=i + 1,
            col=1,
        )

        # Add test data
        fig_samples.add_trace(
            go.Scatter(
                x=test_data["ds"],
                y=test_data["y"],
                mode="lines+markers",
                name=f"{series_id} (Test)",
                line=dict(color="green"),
                legendgroup=series_id,
                showlegend=(i == 0),
            ),
            row=i + 1,
            col=1,
        )

    fig_samples.update_layout(
        height=300 * len(sample_series),
        title_text="Train/Test Split for Sample Series",
        template="plotly_white",
    )

    samples_html = fig_samples.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Sample Series with Train/Test Split</h2>
                {samples_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Visualize how historical data is split into training and testing sets for model evaluation.</p>
                </div>
            </div>
    """)

    # Close HTML document
    html_parts.append("""
        </div>
    </body>
    </html>
    """)

    # Combine all HTML parts
    complete_html = "".join(html_parts)

    # Return as HTMLString
    return HTMLString(complete_html)
