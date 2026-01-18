"""
Парсер объявлений с Kufar для синтезаторов и пианино.
"""

import time
import re
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
from urllib.parse import urlencode, unquote
from base64 import b64decode, b64encode

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser


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
            # Ищем cursor в ссылках
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
                            if isinstance(cursor_data, dict) and cursor_data.get('p', 0) >= 2:
                                return cursor
                        except:
                            if len(cursor) > 20:
                                return cursor
            
            # Если cursor не найден в ссылках, строим его из последнего item ID
            item_links = soup.find_all('a', href=re.compile(r'/item/\d+'))
            if item_links:
                last_href = item_links[-1].get('href', '')
                item_match = re.search(r'/item/(\d+)', last_href)
                if item_match:
                    cursor_data = {
                        "t": "abs",
                        "f": True,
                        "p": 2,
                        "pit": item_match.group(1)
                    }
                    cursor_json = json.dumps(cursor_data, separators=(',', ':'))
                    cursor_b64 = b64encode(cursor_json.encode()).decode('utf-8')
                    return unquote(cursor_b64)
            
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
