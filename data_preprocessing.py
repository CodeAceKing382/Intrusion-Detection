import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data_from_folders(data_folder):
    data_dict = {}
    column_check_dict = {}

    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)

        if os.path.isdir(label_folder):
            df_list = []

            for csv_file in os.listdir(label_folder):
                csv_check_dict = {}
                base_columns = None
                if csv_file.endswith(".csv"):
                    csv_path = os.path.join(label_folder, csv_file)

                    df = pd.read_csv(csv_path)
                    df['label'] = label
                    df_list.append(df)
                    current_columns = df.columns

                    if base_columns is None:
                        base_columns = set(current_columns)
                    csv_check_dict[csv_file] = (set(current_columns) == base_columns)
                column_check_dict[label] = csv_check_dict

            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)
                data_dict[label] = combined_df

    return data_dict, column_check_dict

def replace_infs_with_nans(df):
    return df.replace([np.inf, -np.inf], np.nan)

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Unknown strategy. Use 'mean', 'median', or 'drop'.")

def _generate_features(df):
    """
    Engineers new features from a given Pandas DataFrame for intrusion detection.

    Args:
        df: The input DataFrame containing network traffic data.

    Returns:
        pandas.DataFrame: A DataFrame with the new engineered features and some
                          redundant features removed.  Returns None if an error occurs.
    """

    try:
        # 1. SYN/ACK Ratio
        df['SYN_ACK_Ratio'] = df['syn_flag_number'] / (df['ack_flag_number'] + 1e-6)

        # 2. RST/FIN Ratio
        df['RST_FIN_Ratio'] = df['rst_flag_number'] / (df['fin_flag_number'] + 1e-6)

        # 3. Average Packet Size
        df['Avg_Packet_Size'] = df['Tot size'] / df['Number']

        # 4. Protocol HTTP or HTTPS
        df['HTTP_or_HTTPS'] = ((df['HTTP'] == 1) | (df['HTTPS'] == 1)).astype(int)

        # 5. Non-HTTP/HTTPS Traffic
        df['Non_HTTP_HTTPS'] = ((df['HTTP'] == 0) & (df['HTTPS'] == 0)).astype(int)

        # 6. High Rate Flag (requires threshold determination)
        high_rate_threshold = df['Rate'].quantile(0.95)  # Example: 95th percentile
        df['High_Rate_Flag'] = (df['Rate'] > high_rate_threshold).astype(int)


        # 7. Small Packet Flag (requires threshold determination)
        small_packet_threshold = df['Avg_Packet_Size'].quantile(0.05)  # Example: 5th percentile
        df['Small_Packet_Flag'] = (df['Avg_Packet_Size'] < small_packet_threshold).astype(int)

        # 8. Combined Flags
        flag_cols = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
                     'ack_flag_number', 'ece_flag_number', 'cwr_flag_number']
        df['Combined_Flags'] = df[flag_cols].sum(axis=1)


        # 1. Derived Features from Flags
        df['Flag_Combination_Index'] = (df['ack_count'] * 1.5 + df['syn_count'] * 2 + 
                                        df['fin_count'] * 1 + df['rst_count'] * 1.2)
        df['Flag_Sum'] = df['ack_count'] + df['syn_count'] + df['fin_count'] + df['rst_count']
        df['Flag_Diversity'] = df[['ack_count', 'syn_count', 'fin_count', 'rst_count']].gt(0).sum(axis=1)

        df['TTL_Variance'] = df.groupby('Protocol Type')['Time_To_Live'].transform('var')

        # 6. Feature Interactions
        df['Protocol_vs_Flag_Interaction'] = df['Protocol Type'] * df['syn_count']


        # 11. Ratio Features
        df['Payload_Header_Ratio'] = df['Tot size'] / (df['Header_Length'] + 1e-6)



        # 9. Time To Live Anomaly (requires TTL range determination)
        ttl_lower_bound = df['Time_To_Live'].quantile(0.025) #Example 2.5th percentile
        ttl_upper_bound = df['Time_To_Live'].quantile(0.975) #Example 97.5th percentile

        df['TTL_Anomaly'] = (
            (df['Time_To_Live'] < ttl_lower_bound) | (df['Time_To_Live'] > ttl_upper_bound)
        ).astype(int)


        # Drop potentially redundant features
        cols_to_drop = ['HTTP', 'HTTPS', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
                       'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number']

        df = df.drop(columns=cols_to_drop, errors='ignore')  # errors='ignore' in case you've already dropped some

        return df


    except (KeyError, TypeError, ValueError) as e:
        print(f"Error during feature generation: {e}")
        return None
    
def load_and_preprocess_data(classification_type):
    print("....DATA PREPROCESSING STARTED....")
    data_dict, csv_column_checks = load_data_from_folders('data')
    data = pd.concat([df for df in data_dict.values()], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data = replace_infs_with_nans(data) 
    data.dropna(inplace=True)
    data = _generate_features(data)

    if (classification_type=='b'):
        data['label'] = (data['label'] == 'DDoS').astype(int)
    
    else:
        le = LabelEncoder()
        data['label'] = le.fit_transform(data['label'])


    X = data.drop(columns=['label'])
    y = data['label']

    X = replace_infs_with_nans(X)
    X = handle_missing_values(X, strategy='mean')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test) 
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    print("....DATA PREPROCESSING COMPLETED....")
    
    return X_train_scaled, X_test_scaled,X_val_scaled, y_train, y_test ,y_val

