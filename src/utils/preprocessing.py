"""Preprocessing pipeline for the bank marketing dataset.

Builds a scikit-learn Pipeline that handles:
- Numeric features: median imputation + StandardScaler
- Categorical features: constant imputation + OneHotEncoder
"""

from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric column names present in the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        List of numeric column names.
    """
    return df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Return list of categorical (object/string) column names present in the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        List of categorical column names.
    """
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def build_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer-based preprocessing pipeline.

    Numeric pipeline: SimpleImputer(median) → StandardScaler.
    Categorical pipeline: SimpleImputer(constant='missing') → OneHotEncoder.

    The OneHotEncoder uses handle_unknown='ignore' so that unseen categories
    during prediction do not cause errors.

    Args:
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.

    Returns:
        A fitted or unfitted ColumnTransformer ready to be used in a Pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",  # Drop any column not listed
    )

    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: A fitted ColumnTransformer with numeric and categorical steps.

    Returns:
        List of all output feature names after transformation.
    """
    names: List[str] = []

    for transformer_name, _, cols in preprocessor.transformers_:
        if transformer_name == "num":
            names.extend(cols)
        elif transformer_name == "cat":
            # Get feature names from the fitted OneHotEncoder
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            if hasattr(ohe, "get_feature_names_out"):
                names.extend(ohe.get_feature_names_out(cols).tolist())
            else:
                # Fallback for older sklearn versions
                for col in cols:
                    names.append(f"{col}_encoded")
        elif transformer_name == "remainder":
            pass

    return names
