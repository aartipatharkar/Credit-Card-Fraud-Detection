def preprocess_data(df):

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Fill missing values
    df = df.fillna(0)

    return df
