import sqlite3
import joblib
import pandas as pd

CURRENT_YEAR = 2026

FEATURES = [
    "Name",
    "SubName",
    "Age",
    "Keys",
    "HammerAction",
    "VelocitySensitive",
    "Timbres",
]


def map_unknown_subname_to_other(model, df: pd.DataFrame, col: str = "SubName") -> pd.DataFrame:
    """Маппит незнакомые категории SubName в OTHER согласно OneHotEncoder внутри сохранённого Pipeline."""
    df = df.copy()

    preprocess = model.named_steps["preprocess"]
    ohe = preprocess.named_transformers_["cat"]  # здесь cat — это OneHotEncoder

    # column 1 = SubName (column 0 обычно Name/Brand)
    known_categories = set(ohe.categories_[1])

    df[col] = df[col].astype(str).str.upper().str.strip()
    df.loc[~df[col].isin(known_categories), col] = "OTHER"

    return df


def load_enriched_listings(db_path: str = "keyscout.db") -> pd.DataFrame:
    """Загружает данные для скоринга из таблицы listings_enriched."""
    query = """
        SELECT
            price,
            Name,
            SubName,
            YearIntroduced,
            Keys,
            HammerAction,
            VelocitySensitive,
            Timbres
        FROM listings_enriched;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def prepare_features_for_scoring(df: pd.DataFrame, current_year: int = CURRENT_YEAR, subname_min_count: int = 3) -> pd.DataFrame:
    """Подготавливает фичи: чистит NaN, делает Age, группирует редкие SubName в OTHER."""
    df = df.copy()

    # убираем строки с пропусками (как у тебя)
    df.dropna(inplace=True)

    # группируем редкие SubName в OTHER
    vc = df["SubName"].value_counts()
    rare = vc[vc < subname_min_count].index
    df["SubName"] = df["SubName"].replace(rare, "OTHER")

    # возраст
    df["Age"] = current_year - df["YearIntroduced"]

    # важно: drop должен быть с inplace=True или с присваиванием
    df.drop(columns=["YearIntroduced"], inplace=True)

    return df


def predict_market_price(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Загружает модель, маппит unknown SubName -> OTHER, делает предсказание market_price."""
    model = joblib.load(model_path)

    df = map_unknown_subname_to_other(model, df, col="SubName")

    X = df[FEATURES]
    df["market_price"] = model.predict(X)

    return df


if __name__ == "__main__":
    # 1) Загружаем обогащённые объявления из SQLite
    df = load_enriched_listings(db_path="keyscout.db")

    # 2) Готовим фичи (без изменения твоей логики)
    df = prepare_features_for_scoring(df, current_year=CURRENT_YEAR, subname_min_count=3)

    # 3) Скорим моделью и добавляем market_price
    df = predict_market_price(df, model_path="models/SubName+OTHERS/price_model_market.joblib")

    # df теперь содержит price и market_price (и все фичи)
    print(df[["price", "market_price"]].head())
