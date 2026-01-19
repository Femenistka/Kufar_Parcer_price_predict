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
from typing import Iterable


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

    def _ensure_name_columns(self) -> None:
        """Добавляет колонки Name/SubName в listings, если их нет."""
        cursor = self.conn.cursor()

        cursor.execute("PRAGMA table_info(listings)")
        cols = {row[1] for row in cursor.fetchall()}

        if "Name" not in cols:
            cursor.execute("ALTER TABLE listings ADD COLUMN Name TEXT")
        if "SubName" not in cols:
            cursor.execute("ALTER TABLE listings ADD COLUMN SubName TEXT")

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
    
    def normalize_titles_for_ids(self, ids: Iterable[int]) -> int:
        """Нормализовать Name/SubName для конкретных listings.id"""
        self._ensure_name_columns()
        cursor = self.conn.cursor()

        # забираем title только для указанных id
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
            if name is None and subname is None:
                continue
            to_update.append((name, subname, row_id))

        if to_update:
            cursor.executemany(
                "UPDATE listings SET Name=?, SubName=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                to_update
            )
            self.conn.commit()

        return len(to_update)



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
