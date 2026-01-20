"""
Парсер объявлений с Kufar для синтезаторов и пианино.
"""

import time
import json
from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta
from urllib.parse import urlencode, unquote
from base64 import b64decode, b64encode
import re
import unicodedata
import sqlite3
from typing import Optional, Tuple
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
import pandas as pd
import joblib
import numpy as np

@dataclass
class ListingRaw:
    """Структура сырых данных объявления."""
    source_id: Optional[str] = None
    url: str = ""
    title: str = ""
    price: Optional[float] = None
    currency: str = "BYN"
    published_at: Optional[datetime] = None
    location: str = ""
    description: str = ""
    raw_text: str = ""


class Database:
    """Класс для работы с SQLite БД."""
    
    def __init__(self, db_path: str = "keyscout.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()
    
    def _create_tables(self):
        """Создание таблиц в БД."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                price REAL,
                currency TEXT DEFAULT 'BYN',
                published_at TEXT,
                location TEXT,
                description TEXT,
                raw_text TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_id ON listings(source_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_url ON listings(url)
        """)
        
        self.conn.commit()
    
    def save_listing(self, listing: ListingRaw) -> bool:
        """Сохранение или обновление объявления в БД."""
        try:
            cursor = self.conn.cursor()
            
            published_at_str = listing.published_at.isoformat() if listing.published_at else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO listings 
                (source_id, url, title, price, currency, published_at, location, description, raw_text, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                listing.source_id,
                listing.url,
                listing.title,
                listing.price,
                listing.currency,
                published_at_str,
                listing.location,
                listing.description,
                listing.raw_text
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Ошибка при сохранении объявления: {e}")
            self.conn.rollback()
            return False
    
    def save_listings(self, listings: List[ListingRaw]) -> int:
        """Сохранение списка объявлений."""
        saved_count = 0
        for listing in listings:
            if self.save_listing(listing):
                saved_count += 1
        return saved_count
    
    def close(self):
        """Закрытие соединения с БД."""
        self.conn.close()

    def _ensure_model_columns(self) -> None:
        """Добавляет колонки Name/SubName/IndexModel в listings, если их нет."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(listings)")
        cols = {row[1] for row in cursor.fetchall()}

        if "Name" not in cols:
            cursor.execute("ALTER TABLE listings ADD COLUMN Name TEXT")
        if "SubName" not in cols:
            cursor.execute("ALTER TABLE listings ADD COLUMN SubName TEXT")
        if "IndexModel" not in cols:
            cursor.execute("ALTER TABLE listings ADD COLUMN IndexModel TEXT")
        if "IndexModelInt" not in cols:
            cursor.execute("ALTER TABLE listings ADD COLUMN IndexModelInt INTEGER")

        self.conn.commit()

    @staticmethod
    def _normalize_title_text(s: Optional[str]) -> str:
        """Нормализация строки: Unicode, пробелы, верхний регистр."""
        if not s:
            return ""

        # NFKC + убрать диакритику
        s = unicodedata.normalize("NFKC", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

        # привести разные дефисы к "-"
        s = s.replace("–", "-").replace("—", "-").replace("−", "-")

        # upper + трим + схлопнуть пробелы
        s = s.upper().strip()
        s = re.sub(r"\s+", " ", s)

        return s

    @staticmethod
    def _apply_synonyms(s: str) -> str:
        """
        Заменяет русские/транслит варианты на каноничные токены.
        Делается ПОСЛЕ upper().
        """
        if not s:
            return s

        # 1) бренды
        brand_patterns = [
            (r"\b(КАСИО|КЭ?СИО|CASIO)\b", "CASIO"),
            (r"\b(ЯМАХА|YAMAHA)\b", "YAMAHA"),
        ]

        # 2) серии / линейки (русские варианты тоже)
        series_patterns = [
            (r"\b(ПСР|PSR)\b", "PSR"),
            # "P" важно заменить аккуратно (часто встречается как "P-45", "P 125")
            (r"\bP(?=\s*[-]?\s*\d)", "P"),
            (r"\b(DGX)\b", "DGX"),
            (r"\b(CLP)\b", "CLP"),
            (r"\b(YDP)\b", "YDP"),
            (r"\b(NP)\b", "NP"),
            (r"\b(MX)\b", "MX"),
            (r"\b(DX)\b", "DX"),
            (r"\b(CS)\b", "CS"),

            (r"\b(CDP)\b", "CDP"),
            (r"\b(PX)\b", "PX"),
            (r"\b(AP)\b", "AP"),
            (r"\b(CTK)\b", "CTK"),
            (r"\b(CT\s*-\s*S)\b", "CT-S"),
            (r"\b(CT\s*-\s*X)\b", "CT-X"),
            (r"\b(WK)\b", "WK"),
            (r"\b(SA)\b", "SA"),
            (r"\b(LK)\b", "LK"),
        ]

        # 3) дополнительные чистки: "CELVIANO" / "CLAVINOVA" — не subname, но полезно оставить (не мешает)
        misc_patterns = [
            (r"\b(CELVIANO)\b", "CELVIANO"),
            (r"\b(CLAVINOVA)\b", "CLAVINOVA"),
            (r"\b(PIAGGERO)\b", "PIAGGERO"),
        ]

        for pat, repl in brand_patterns:
            s = re.sub(pat, repl, s)

        for pat, repl in series_patterns:
            s = re.sub(pat, repl, s)

        for pat, repl in misc_patterns:
            s = re.sub(pat, repl, s)

        # почистим пробелы вокруг дефисов (P - 45 -> P-45, CT - S -> CT-S)
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"\s+", " ", s).strip()
        # Нормализация типичных форматов:
        # PSR E-463 -> PSR-E463, PSR E463 -> PSR-E463
        s = re.sub(r"\bPSR\s+E\s*-\s*(\d+)\b", r"PSR-E\1", s)
        s = re.sub(r"\bPSR\s+E(\d+)\b", r"PSR-E\1", s)

        # CDP 120BK / CDP-120BK -> CDP-120BK (чтобы CDP точно было видно)
        s = re.sub(r"\bCDP\s*-\s*(S?\d+)\b", r"CDP-\1", s)
        s = re.sub(r"\bCDP\s+(\d+[A-Z]*)\b", r"CDP-\1", s)

        # NP32 -> NP32 (ничего не меняем, но гарантируем верхний регистр уже есть)


        return s

    @staticmethod
    def _extract_name_subname(s: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Извлекает:
        - Name: CASIO / YAMAHA (или выводит по серии, если бренд не указан)
        - SubName: PSR / P / DGX / CLP / YDP / NP / CDP / CTK / CT-S / CT-X / PX / AP / WK / SA / LK / MX / DX / CS
        Работает с вариантами типа CT-X800, NP32, CLP-535WH (где серия прилеплена к цифрам/буквам).
        """
        if not s:
            return None, None

        # 1) BRAND (явно)
        name = None
        if re.search(r"\bCASIO\b", s):
            name = "CASIO"
        elif re.search(r"\bYAMAHA\b", s):
            name = "YAMAHA"

        # 2) SUBNAME: ищем токены как "слово" ИЛИ как префикс перед цифрами/буквами
        def has_token(token: str) -> bool:
            # token как отдельное слово, или token перед цифрой/буквой (CT-X800 / NP32 / CLP-535WH)
            return re.search(rf"(?:\b{re.escape(token)}\b)|(?:\b{re.escape(token)}(?=[0-9A-Z]))", s) is not None

        # порядок важен: более специфичное раньше
        subname_order = [
            "PSR", "DGX", "CLP", "YDP", "NP", "MX", "DX", "CS",
            "CDP", "PX", "AP", "CT-S", "CT-X", "CTK", "WK", "SA", "LK",
            "P",  # P ставим ближе к концу, чтобы не перехватывал лишнее
        ]

        subname = None
        for token in subname_order:
            if has_token(token):
                subname = token
                break

        # 3) Спец-кейсы для Yamaha P-серии (P-45 / P 125 / P125)
        if subname is None and re.search(r"\bP\s*-\s*\d+|\bP\s+\d+|\bP\d+\b", s):
            subname = "P"

        # 4) Если бренд НЕ указан — выводим бренд по серии (очень полезно для строк типа "Синтезатор PSR E-463")
        if name is None and subname is not None:
            yamaha_series = {"PSR", "P", "DGX", "CLP", "YDP", "NP", "MX", "DX", "CS"}
            casio_series = {"CDP", "PX", "AP", "CT-S", "CT-X", "CTK", "WK", "SA", "LK"}

            if subname in yamaha_series:
                name = "YAMAHA"
            elif subname in casio_series:
                name = "CASIO"

        return name, subname

    def normalize_titles_to_name_subname(self, batch_size: int = 500) -> int:
        """
        Обходит listings, парсит title -> (Name, SubName) и записывает в БД.
        Возвращает количество обновлённых строк.
        """
        self._ensure_name_columns()

        cursor = self.conn.cursor()

        cursor.execute("SELECT id, title FROM listings WHERE title IS NOT NULL")
        rows = cursor.fetchall()

        updated = 0
        to_update = []

        for row_id, title in rows:
            t = self._normalize_title_text(title)
            t = self._apply_synonyms(t)
            name, subname = self._extract_name_subname(t)

            # записываем только если есть хоть что-то
            if name is None and subname is None:
                continue

            to_update.append((name, subname, row_id))

            if len(to_update) >= batch_size:
                cursor.executemany(
                    "UPDATE listings SET Name = ?, SubName = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    to_update
                )
                self.conn.commit()
                updated += len(to_update)
                to_update = []

        if to_update:
            cursor.executemany(
                "UPDATE listings SET Name = ?, SubName = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                to_update
            )
            self.conn.commit()
            updated += len(to_update)

        return updated

    def save_listing_return_id(self, listing: ListingRaw) -> int | None:
        """Сохранить/обновить и вернуть id строки."""
        try:
            cursor = self.conn.cursor()
            published_at_str = listing.published_at.isoformat() if listing.published_at else None

            cursor.execute("""
                INSERT INTO listings (source_id, url, title, price, currency, published_at, location, description, raw_text, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source_id) DO UPDATE SET
                    url=excluded.url,
                    title=excluded.title,
                    price=excluded.price,
                    currency=excluded.currency,
                    published_at=excluded.published_at,
                    location=excluded.location,
                    description=excluded.description,
                    raw_text=excluded.raw_text,
                    updated_at=CURRENT_TIMESTAMP
            """, (
                listing.source_id, listing.url, listing.title, listing.price, listing.currency,
                published_at_str, listing.location, listing.description, listing.raw_text
            ))

            self.conn.commit()

            cursor.execute("SELECT id FROM listings WHERE source_id = ?", (listing.source_id,))
            row = cursor.fetchone()
            return int(row[0]) if row else None

        except Exception as e:
            print(f"Ошибка при сохранении объявления: {e}")
            self.conn.rollback()
            return None

    def save_listings_return_ids(self, listings: list[ListingRaw]) -> list[int]:
        ids: list[int] = []
        for l in listings:
            row_id = self.save_listing_return_id(l)
            if row_id is not None:
                ids.append(row_id)
        return ids
    
    def normalize_titles_for_ids(self, ids):
        self._ensure_model_columns()
        cursor = self.conn.cursor()

        ids = list(ids)
        if not ids:
            return 0

        placeholders = ",".join(["?"] * len(ids))
        cursor.execute(f"SELECT id, title FROM listings WHERE id IN ({placeholders})", ids)
        rows = cursor.fetchall()

        to_update = []
        for row_id, title in rows:
            t = self._normalize_title_text(title)
            t = self._apply_synonyms(t)

            name, subname = self._extract_name_subname(t)
            index_model = self._extract_index_model(t, name, subname)

            # если вообще ничего не нашли — пропускаем
            if name is None and subname is None and index_model is None:
                continue

            to_update.append((name, subname, index_model, row_id))

        if to_update:
            cursor.executemany(
                """
                UPDATE listings
                SET
                Name = COALESCE(?, Name),
                SubName = COALESCE(?, SubName),
                IndexModel = COALESCE(?, IndexModel),
                updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                to_update
            )
            self.conn.commit()

        return len(to_update)

    def ensure_enriched_prediction_column(self, table: str = "listings_enriched") -> None:
            """Добавляет price_predict в таблицу listings_enriched, если его нет."""
            cur = self.conn.cursor()
            cur.execute(f"PRAGMA table_info({table})")
            cols = {row[1] for row in cur.fetchall()}

            if "price_predict" not in cols:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN price_predict REAL")
            if "market_price" not in cols:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN market_price REAL")

            self.conn.commit()

    @staticmethod
    def _extract_index_model(normalized_title: str, name: Optional[str], subname: Optional[str]) -> Optional[str]:
        """
        Грубый MVP-метод:
        - убираем из строки Name/SubName
        - ищем группы цифр (2+ цифры)
        - берём "самую вероятную": сначала 3-4 цифры, потом 2 цифры
        """
        if not normalized_title:
            return None

        s = normalized_title

        # 1) выкинуть бренд/серию, если известны
        if name:
            s = re.sub(rf"\b{re.escape(name)}\b", " ", s)
        if subname:
            # subname может быть CT-S, CT-X => обязательно escape
            s = re.sub(rf"\b{re.escape(subname)}\b", " ", s)

        # 2) почистить популярный шум (опционально, но помогает)
        # (можно расширять)
        s = re.sub(r"\b(PIANO|PIANINO|SYNTH|SYNTHESIZER|СИНТЕЗАТОР|ПИАНИНО|ЦИФРОВОЕ|ЭЛЕКТРОННОЕ)\b", " ", s)
        s = re.sub(r"\b(BK|WH|SR|WE|B|BLACK|WHITE)\b", " ", s)

        # 3) привести разделители к пробелам, чтобы легче ловить цифры
        s = s.replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()

        if not s:
            return None

        # 4) сначала ищем "классические индексы" 3-4 цифры: 100, 270, 463, 164, 535, 620, 775...
        m = re.search(r"\b(\d{3,4})\b", s)
        if m:
            return m.group(1)

        # 5) если не нашли — берём 2 цифры (NP32, MX61, SA76, F51...)
        m = re.search(r"\b(\d{2})\b", s)
        if m:
            return m.group(1)

        # 6) крайний случай: вообще любые цифры
        m = re.search(r"(\d+)", s)
        return m.group(1) if m else None

    def load_model_specs_csv(
        self,
        csv_path: str,
        table_name: str = "model_specs",
        if_exists: str = "replace"
    ) -> int:
        """
        Загружает CSV с характеристиками в SQLite таблицу model_specs.
        Требует колонки: Name, SubName, IndexModel + любые характеристики.
        Возвращает число строк.
        """
        df = pd.read_csv(csv_path)

        # убрать мусорные индексы
        for col in ["Unnamed: 0", "unnamed: 0"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        required = {"Name", "SubName", "IndexModel"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV не содержит обязательные колонки: {sorted(missing)}")

        # нормализуем ключи
        df["Name"] = df["Name"].astype(str).str.upper().str.strip()
        df["SubName"] = df["SubName"].astype(str).str.upper().str.strip()

        # IndexModel -> int (если не получается, будет <NA>)
        df["IndexModelInt"] = pd.to_numeric(df["IndexModel"], errors="coerce").astype("Int64")

        # bool -> 0/1 (удобно для ML и SQLite)
        for bcol in ["HammerAction", "VelocitySensitive"]:
            if bcol in df.columns:
                # если уже bool или строки True/False
                df[bcol] = df[bcol].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(df[bcol])
                df[bcol] = pd.to_numeric(df[bcol], errors="coerce").astype("Int64")

        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)

        cur = self.conn.cursor()
        # индексы для быстрого мэтчинга
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_name_sub ON {table_name}(Name, SubName)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_name_sub_idx ON {table_name}(Name, SubName, IndexModelInt)")
        self.conn.commit()

        return len(df)
    
    def build_enriched_listings_table(
        self,
        specs_table: str = "model_specs",
        out_table: str = "listings_enriched",
        only_with_keys: bool = True
    ) -> int:
        """
        Создаёт таблицу out_table как результат JOIN:
        listings (Name/SubName/IndexModelInt)
            + model_specs (Name/SubName/IndexModelInt)
        Правило мэтчинга:
        - Name/SubName: совпадение без регистра (мы храним uppercase)
        - IndexModelInt: точное, иначе ближайшее по |diff|
        Возвращает число строк в out_table.
        """
        self._ensure_model_columns()
        cur = self.conn.cursor()

        # Если IndexModelInt ещё не заполнен для listings — попробуем заполнить из IndexModel (если оно числовое)
        cur.execute("""
            UPDATE listings
            SET IndexModelInt = CASE
                WHEN IndexModelInt IS NULL THEN CAST(IndexModel AS INTEGER)
                ELSE IndexModelInt
            END
            WHERE IndexModel IS NOT NULL
        """)
        self.conn.commit()

        # Получаем колонки из таблицы характеристик, чтобы динамически их притащить
        cur.execute(f"PRAGMA table_info({specs_table})")
        spec_cols = [row[1] for row in cur.fetchall()]
        # ключевые колонки спецификаций
        key_cols = {"Name", "SubName", "IndexModel", "IndexModelInt"}
        feature_cols = [c for c in spec_cols if c not in key_cols]

        # пересоздаём итоговую таблицу
        cur.execute(f"DROP TABLE IF EXISTS {out_table}")

        # собираем SELECT-часть
        select_features = ", ".join([f"best.{c} AS {c}" for c in feature_cols]) if feature_cols else ""
        if select_features:
            select_features = ", " + select_features

        # optionally фильтр: только те listings, где уже есть ключи
        where_clause = ""
        if only_with_keys:
            where_clause = "WHERE l.Name IS NOT NULL AND l.SubName IS NOT NULL AND l.IndexModelInt IS NOT NULL"

        # ВАЖНО: window function row_number() выберет самый близкий IndexModelInt
        sql = f"""
        CREATE TABLE {out_table} AS
        WITH candidates AS (
            SELECT
                l.id AS listing_id,
                s.*,
                ABS(s.IndexModelInt - l.IndexModelInt) AS dist,
                ROW_NUMBER() OVER (
                    PARTITION BY l.id
                    ORDER BY ABS(s.IndexModelInt - l.IndexModelInt) ASC
                ) AS rn
            FROM listings l
            JOIN {specs_table} s
            ON l.Name = s.Name
            AND l.SubName = s.SubName
            WHERE l.IndexModelInt IS NOT NULL
            AND s.IndexModelInt IS NOT NULL
        ),
        best AS (
            SELECT * FROM candidates WHERE rn = 1
        )
        SELECT
            l.id AS listing_id,
            l.source_id,
            l.description,
            l.raw_text,
            l.url,
            l.title,
            l.price,
            l.currency,
            l.published_at,
            l.location,
            l.created_at,
            l.updated_at,
            l.Name,
            l.SubName,
            l.IndexModel,
            l.IndexModelInt,
            best.IndexModelInt AS MatchedIndexModelInt,
            best.dist AS IndexModelDistance
            {select_features}
        FROM listings l
        LEFT JOIN best ON best.listing_id = l.id
        {where_clause}
        ;
        """

        cur.execute(sql)

        # индексы на итоговую таблицу
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{out_table}_listing_id ON {out_table}(listing_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{out_table}_name_sub_idx ON {out_table}(Name, SubName, IndexModelInt)")
        self.conn.commit()

        cur.execute(f"SELECT COUNT(*) FROM {out_table}")
        return int(cur.fetchone()[0])

    def update_price_predictions(
        self,
        df_pred: pd.DataFrame,
        table: str = "listings_enriched",
        id_col: str = "listing_id",
        market_col: str = "market_price",
    ) -> int:
        """
        Обновляет в listings_enriched значения market_price и price_predict
        по ключу listing_id.
        price_predict = round(market_price, -1) (до десятков)
        """
        self.ensure_enriched_prediction_column(table)

        if id_col not in df_pred.columns or market_col not in df_pred.columns:
            raise ValueError(f"Нужны колонки {id_col} и {market_col} в df_pred")

        tmp = df_pred[[id_col, market_col]].copy()
        tmp[market_col] = pd.to_numeric(tmp[market_col], errors="coerce")
        tmp.dropna(inplace=True)

        # округление до десятков
        tmp["price_predict"] = tmp[market_col].round(-1)

        rows = list(tmp[[id_col, market_col, "price_predict"]].itertuples(index=False, name=None))
        if not rows:
            return 0

        cur = self.conn.cursor()
        cur.executemany(
            f"""
            UPDATE {table}
            SET
              market_price = ?,
              price_predict = ?,
              updated_at = CURRENT_TIMESTAMP
            WHERE {id_col} = ?
            """,
            [(market, pred, lid) for (lid, market, pred) in rows]
        )
        self.conn.commit()
        return cur.rowcount

    def fetch_enriched_for_scoring(self) -> pd.DataFrame:
        query = """
            SELECT
                listing_id,
                price,
                Name,
                SubName,
                YearIntroduced,
                Keys,
                HammerAction,
                VelocitySensitive,
                Timbres
            FROM listings_enriched
            WHERE
                price IS NOT NULL
                AND Name IS NOT NULL
                AND SubName IS NOT NULL
                AND YearIntroduced IS NOT NULL
                AND Keys IS NOT NULL
                AND HammerAction IS NOT NULL
                AND VelocitySensitive IS NOT NULL
                AND Timbres IS NOT NULL
        """
        return pd.read_sql_query(query, self.conn)

    def run_scoring_and_save_predictions(
        self,
        model_path: str,
        current_year: int = 2026,
        subname_min_count: int = 3,
    ) -> dict:
        """
        1) читает listings_enriched
        2) готовит фичи
        3) делает предикт
        4) пишет market_price и price_predict обратно в listings_enriched
        """
        df = self.fetch_enriched_for_scoring()
        if df.empty:
            return {"scored": 0, "updated": 0}

        # ---- подготовка фич (логика как в твоём model.py) ----
        df = df.dropna().copy()

        vc = df["SubName"].value_counts()
        rare = vc[vc < subname_min_count].index
        df["SubName"] = df["SubName"].replace(rare, "OTHER")

        df["Age"] = current_year - df["YearIntroduced"]
        df.drop(columns=["YearIntroduced"], inplace=True)

        # ---- модель ----
        model = joblib.load(model_path)

        # маппинг unknown SubName -> OTHER по энкодеру
        preprocess = model.named_steps["preprocess"]
        ohe = preprocess.named_transformers_["cat"]
        known = set(ohe.categories_[1])
        df["SubName"] = df["SubName"].astype(str).str.upper().str.strip()
        df.loc[~df["SubName"].isin(known), "SubName"] = "OTHER"

        FEATURES = ["Name", "SubName", "Age", "Keys", "HammerAction", "VelocitySensitive", "Timbres"]
        # Перманентное округление предсказаний до десятков 
        df["market_price"] = np.round(model.predict(df[FEATURES]), -1)

        updated = self.update_price_predictions(df_pred=df, id_col="listing_id", market_col="market_price")
        return {"scored": int(len(df)), "updated": int(updated)}

class KufarScraper:
    """Парсер объявлений с Kufar."""
    
    BASE_URL = "https://www.kufar.by"
    
    def __init__(self, delay: float = 1.0, timeout: int = 10):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def build_search_url(self, region: str = "minsk", category: str = "klavishnye",
                        cursor: Optional[str] = None, **kwargs) -> str:
        """Построение URL для поиска на Kufar."""
        base_path = f"/l/r~{region}/{category}/bez-posrednikov"
        
        if cursor is None:
            cursor = 'eyJ0IjoiYWJzIiwiZiI6dHJ1ZSwicCI6MX0%3D'
        
        params = kwargs.copy()
        params.setdefault('cnd', '1')
        params.setdefault('sort', 'lst.d')
        
        query_parts = [f"cnd={params['cnd']}", f"cursor={cursor}"]
        
        if 'mkb' in params:
            query_parts.append(f"mkb={urlencode({'mkb': params['mkb']}, doseq=True).split('=')[1]}")
        if 'mki' in params:
            query_parts.append(f"mki={urlencode({'mki': params['mki']}, doseq=True).split('=')[1]}")
        if 'sort' in params:
            query_parts.append(f"sort={params['sort']}")
        
        return f"{self.BASE_URL}{base_path}?{'&'.join(query_parts)}"
    
    def extract_next_cursor(self, soup: BeautifulSoup) -> Optional[str]:
        """Извлечение cursor для следующей страницы из HTML."""
        try:
            # Ищем cursor только в явных ссылках на следующую страницу
            # Проверяем все ссылки с cursor в href
            found_cursors = []
            
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if 'cursor=' in href:
                    match = re.search(r'cursor=([^&"\'>\s]+)', href)
                    if match:
                        cursor = match.group(1)
                        try:
                            decoded = unquote(cursor)
                            decoded_bytes = b64decode(decoded + '==')
                            cursor_data = json.loads(decoded_bytes)
                            # Проверяем, что это действительно следующая страница (p >= 2)
                            if isinstance(cursor_data, dict):
                                page_num = cursor_data.get('p', 0)
                                if page_num >= 2:
                                    found_cursors.append((cursor, page_num))
                        except:
                            # Если не удалось декодировать, но cursor длинный, сохраняем его
                            if len(cursor) > 20:
                                found_cursors.append((cursor, 999))  # Неизвестный номер страницы
            
            # Если нашли cursors, возвращаем тот, который указывает на самую дальнюю страницу
            if found_cursors:
                # Сортируем по номеру страницы и берем максимальный
                found_cursors.sort(key=lambda x: x[1], reverse=True)
                return found_cursors[0][0]
            
            # Если не нашли cursor в ссылках, проверяем наличие кнопок пагинации
            # Ищем элементы с текстом "следующая", "next" и т.д.
            pagination_keywords = ['следующ', 'next', 'далее', 'вперед', 'forward', '>', '→']
            has_next_button = False
            
            for element in soup.find_all(['a', 'button', 'span', 'div']):
                text = element.get_text(strip=True).lower()
                href = element.get('href', '')
                
                # Проверяем текст и наличие cursor в href
                if any(keyword in text for keyword in pagination_keywords):
                    if 'cursor=' in href:
                        match = re.search(r'cursor=([^&"\'>\s]+)', href)
                        if match:
                            return match.group(1)
                    # Если есть aria-label или data-атрибуты, указывающие на следующую страницу
                    aria_label = element.get('aria-label', '').lower()
                    if any(keyword in aria_label for keyword in pagination_keywords):
                        if 'cursor=' in href:
                            match = re.search(r'cursor=([^&"\'>\s]+)', href)
                            if match:
                                return match.group(1)
            
            # Если не нашли явной ссылки, строим cursor из последнего item ID
            # Это нужно для перехода на следующую страницу, когда явных ссылок нет
            item_links = soup.find_all('a', href=re.compile(r'/item/\d+'))
            if item_links:
                last_href = item_links[-1].get('href', '')
                item_match = re.search(r'/item/(\d+)', last_href)
                if item_match:
                    cursor_data = {
                        "t": "abs",
                        "f": True,
                        "p": 2,  # Следующая страница
                        "pit": item_match.group(1)
                    }
                    cursor_json = json.dumps(cursor_data, separators=(',', ':'))
                    cursor_b64 = b64encode(cursor_json.encode()).decode('utf-8')
                    return unquote(cursor_b64)
            
            # Если не нашли объявлений и не нашли cursor, значит это последняя страница
            return None
            
        except Exception:
            return None
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Загрузка и парсинг HTML страницы."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'lxml')
        except requests.RequestException:
            return None
    
    def parse_listing_card(self, card_element) -> Optional[ListingRaw]:
        """Парсинг карточки объявления из HTML."""
        try:
            listing = ListingRaw()
            
            if card_element.name == 'a' and card_element.get('href'):
                listing.url = card_element['href'].split('?')[0]
            else:
                link_elem = card_element.find('a', href=True)
                if link_elem:
                    listing.url = link_elem['href'].split('?')[0]
            
            if not listing.url:
                return None
            
            match = re.search(r'/item/(\d+)', listing.url)
            if match:
                listing.source_id = match.group(1)
            
            price_elem = card_element.find('p', class_='styles_price__aVxZc')
            if price_elem:
                price_span = price_elem.find('span')
                if price_span:
                    price_text = price_span.get_text(strip=True)
                    listing.price = self._normalize_price(price_text)
                    if 'р.' in price_text or 'руб' in price_text.lower():
                        listing.currency = "BYN"
            
            title_elem = card_element.find('h3', class_='styles_title__F3uIe')
            if title_elem:
                listing.title = title_elem.get_text(strip=True)
            
            secondary_elem = card_element.find('div', class_='styles_secondary__MzdEb')
            if secondary_elem:
                location_elem = secondary_elem.find('p', class_='styles_region__qCRbf')
                if location_elem:
                    listing.location = location_elem.get_text(strip=True)
                
                date_span = secondary_elem.find('span')
                if date_span:
                    date_text = date_span.get_text(strip=True)
                    listing.published_at = self._normalize_date(date_text)
            
            listing.raw_text = card_element.get_text(separator=' ', strip=True)
            return listing
            
        except Exception:
            return None
    
    def _normalize_price(self, price_text: str) -> Optional[float]:
        """Нормализация цены из текста в число."""
        cleaned = re.sub(r'[^\d.,]', '', price_text.replace(' ', ''))
        cleaned = cleaned.replace(',', '.')
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def _normalize_date(self, date_text: str) -> Optional[datetime]:
        """Нормализация даты из текста в datetime."""
        if not date_text:
            return None
        
        date_text = date_text.strip()
        now = datetime.now()
        
        if 'сегодня' in date_text.lower() or 'today' in date_text.lower():
            time_match = re.search(r'(\d{1,2}):(\d{2})', date_text)
            if time_match:
                return now.replace(hour=int(time_match.group(1)), minute=int(time_match.group(2)), second=0, microsecond=0)
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if 'вчера' in date_text.lower() or 'yesterday' in date_text.lower():
            yesterday = now - timedelta(days=1)
            time_match = re.search(r'(\d{1,2}):(\d{2})', date_text)
            if time_match:
                return yesterday.replace(hour=int(time_match.group(1)), minute=int(time_match.group(2)), second=0, microsecond=0)
            return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        
        try:
            return date_parser.parse(date_text, fuzzy=True)
        except (ValueError, TypeError):
            return None
    
    def scrape_search_results(self, region: str = "minsk", category: str = "klavishnye",
                             max_pages: int = 1, **kwargs) -> List[ListingRaw]:
        """Парсинг результатов поиска с нескольких страниц."""
        listings = []
        current_cursor = None
        
        for page in range(1, max_pages + 1):
            print(f"[{page}/{max_pages}] Загрузка страницы...")
            
            url = self.build_search_url(region=region, category=category, cursor=current_cursor, **kwargs)
            soup = self.fetch_page(url)
            
            if not soup:
                print(f"[{page}/{max_pages}] Ошибка загрузки")
                break
            
            cards = soup.find_all('a', {'data-testid': 'kufar-ad'})
            
            if not cards:
                print(f"[{page}/{max_pages}] Объявления не найдены")
                break
            
            page_listings = []
            for card in cards:
                listing = self.parse_listing_card(card)
                if listing:
                    page_listings.append(listing)
            
            listings.extend(page_listings)
            print(f"[{page}/{max_pages}] Найдено объявлений: {len(page_listings)}")
            
            if page < max_pages:
                next_cursor = self.extract_next_cursor(soup)
                if next_cursor:
                    current_cursor = next_cursor
                else:
                    print(f"[{page}/{max_pages}] Cursor не найден, завершение")
                    break
                
                time.sleep(self.delay)
        
        return listings


def main():
    """Основная функция для запуска парсера."""
    print("=" * 60)
    print("Парсер Kufar - Клавишные инструменты")
    print("=" * 60)


    
    scraper = KufarScraper(delay=1.0, timeout=10)
    db = Database("keyscout.db")
    
    test_params = {
        'mkb': 'v.or:1,25',
        'mki': 'v.or:1,5'
    }
    
    print("\nНачало парсинга...")
    listings = scraper.scrape_search_results(
        region="minsk",
        category="klavishnye",
        max_pages=2,
        **test_params
    )
    
    print(f"\nСохранение в БД...")
    saved_count = db.save_listings(listings)
    
    print(f"\nЗавершено!")
    print(f"Найдено объявлений: {len(listings)}")
    print(f"Сохранено в БД: {saved_count}")
    
    db.close()
    print("=" * 60)


if __name__ == "__main__":
    main()
