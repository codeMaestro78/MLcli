"""
Preprocess Hotel Bookings Dataset for ML Training
Handles: Missing values, Categorical encoding, Feature selection
"""

import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_hotel_data():
    """Preprocess hotel bookings data for ML models."""

    # Load data
    input_path = Path("data/hotel_bookings_updated_2024.csv")
    output_path = Path("data/hotel_bookings_processed.csv")

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")

    # Select relevant features (drop high-cardinality and date columns)
    columns_to_drop = [
        'reservation_status_date',  # Date column
        'reservation_status',       # Leakage - this indicates if canceled
        'company',                  # Too many missing values (94%)
        'agent',                    # Many missing values
        'country',                  # High cardinality + missing
        'arrival_date_month',       # Will use week number instead
        'city',                     # High cardinality
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"After dropping columns: {df.shape}")

    # Handle missing values
    df['children'] = df['children'].fillna(0)

    # Encode categorical variables
    categorical_cols = [
        'hotel', 'meal', 'market_segment', 'distribution_channel',
        'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type'
    ]

    print("Encoding categorical columns...")
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    # Ensure target is at the end
    target = df['is_canceled']
    df = df.drop('is_canceled', axis=1)
    df['is_canceled'] = target

    # Remove any remaining missing values
    df = df.dropna()
    print(f"Final shape: {df.shape}")

    # Print column info
    print("\nFinal columns:")
    print(df.columns.tolist())
    print(f"\nTarget distribution:")
    print(df['is_canceled'].value_counts())

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")

    return df

if __name__ == "__main__":
    preprocess_hotel_data()
