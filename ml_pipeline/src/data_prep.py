from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(df):
    """
    Splits data into training and test sets and scales the features.
    """
    # Split features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
