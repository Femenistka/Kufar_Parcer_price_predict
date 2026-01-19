"""
Streamlit UI для парсера Kufar.
"""

import streamlit as st
import sqlite3
from datetime import datetime
from typing import List
from scraper import KufarScraper, Database, ListingRaw
import time
from DB_functions import clear_db


# Настройка страницы
st.set_page_config(
    page_title="KeyScout - Парсер Kufar",
    page_icon="🎹",
    layout="wide"
)


def scrape_all_pages(scraper: KufarScraper, region: str = "minsk", 
                     category: str = "klavishnye", **kwargs) -> List[ListingRaw]:
    """Парсинг всех доступных страниц."""
    listings = []
    seen_ids = set()  # Для отслеживания дубликатов
    current_cursor = None
    page = 1
    max_pages = 1000  # Защита от бесконечного цикла
    empty_pages_count = 0  # Счетчик пустых страниц подряд
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while page <= max_pages:
        status_text.text(f"Загрузка страницы {page}...")
        progress_bar.progress(min(page / 100, 1.0))  # Максимум 100 страниц для прогресс-бара
        
        url = scraper.build_search_url(region=region, category=category, 
                                      cursor=current_cursor, **kwargs)
        soup = scraper.fetch_page(url)
        
        if not soup:
            status_text.text(f"Ошибка загрузки страницы {page}")
            empty_pages_count += 1
            if empty_pages_count >= 2:
                status_text.text(f"Две пустые страницы подряд. Завершение парсинга.")
                break
            page += 1
            continue
        
        cards = soup.find_all('a', {'data-testid': 'kufar-ad'})
        
        if not cards or len(cards) == 0:
            status_text.text(f"Объявления не найдены на странице {page}")
            empty_pages_count += 1
            if empty_pages_count >= 2:
                status_text.text(f"Две пустые страницы подряд. Завершение парсинга.")
                break
            page += 1
            continue
        
        empty_pages_count = 0  # Сбрасываем счетчик, если нашли объявления
        
        page_listings = []
        new_listings_count = 0
        
        for card in cards:
            listing = scraper.parse_listing_card(card)
            if listing and listing.source_id:
                # Проверяем на дубликаты
                if listing.source_id not in seen_ids:
                    seen_ids.add(listing.source_id)
                    page_listings.append(listing)
                    new_listings_count += 1
        
        # Если на странице нет новых объявлений, возможно мы зациклились
        if new_listings_count == 0 and len(listings) > 0:
            status_text.text(f"На странице {page} нет новых объявлений. Возможно достигнут конец.")
            break
        
        listings.extend(page_listings)
        status_text.text(f"Страница {page}: найдено {len(page_listings)} новых объявлений. Всего уникальных: {len(listings)}")
        
        # Проверяем, есть ли следующая страница
        next_cursor = scraper.extract_next_cursor(soup)
        if not next_cursor:
            status_text.text(f"Следующая страница не найдена. Всего найдено: {len(listings)} объявлений")
            break
        
        # Проверяем, не повторяется ли cursor (защита от зацикливания)
        if current_cursor == next_cursor:
            status_text.text(f"Cursor не изменился. Завершение парсинга.")
            break
        
        current_cursor = next_cursor
        page += 1
        time.sleep(scraper.delay)
    
    progress_bar.progress(1.0)
    status_text.text(f"Парсинг завершен. Всего найдено уникальных объявлений: {len(listings)}")
    return listings


def format_price(price: float, currency: str) -> str:
    """Форматирование цены для отображения."""
    if price is None:
        return "Цена не указана"
    return f"{price:,.0f} {currency}".replace(",", " ")


def format_date(date: datetime) -> str:
    """Форматирование даты для отображения."""
    if date is None:
        return "Дата не указана"
    return date.strftime("%d.%m.%Y %H:%M")


def display_listing_card(listing_data: dict):
    """Отображение карточки объявления."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### {listing_data['title'] or 'Без названия'}")
        if listing_data['description']:
            st.markdown(f"*{listing_data['description'][:200]}...*" if len(listing_data['description']) > 200 else f"*{listing_data['description']}*")
    
    with col2:
        st.markdown(f"**{format_price(listing_data['price'], listing_data['currency'])}**")
        st.markdown(f"📍 {listing_data['location'] or 'Не указано'}")
        st.markdown(f"📅 {format_date(listing_data['published_at'])}")
    
    if listing_data['url']:
        st.markdown(f"[🔗 Открыть объявление]({listing_data['url']})")
    
    st.divider()


def get_listings_from_db(db_path: str = "keyscout.db") -> List[dict]:
    """Получение всех объявлений из БД."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT source_id, url, title, price, currency, published_at, 
               location, description, raw_text, created_at, updated_at
        FROM listings
        ORDER BY updated_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    listings = []
    for row in rows:
        published_at = None
        if row['published_at']:
            try:
                published_at = datetime.fromisoformat(row['published_at'])
            except:
                pass
        
        listings.append({
            'source_id': row['source_id'],
            'url': row['url'],
            'title': row['title'],
            'price': row['price'],
            'currency': row['currency'] or 'BYN',
            'published_at': published_at,
            'location': row['location'],
            'description': row['description'] or row['raw_text'] or '',
            'raw_text': row['raw_text'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        })
    
    return listings


# Главное меню - вкладки в шапке
tab1, tab2 = st.tabs(["⚙️ Настройки парсинга", "📊 Результаты"])

with tab1:
    st.title("🎹 KeyScout - Парсер объявлений Kufar")
    st.markdown("---")
    
    st.subheader("Настройки парсинга")
    
    # Параметры парсинга
    col1, col2 = st.columns(2)
    
    with col1:
        scrape_all = st.checkbox("Собрать все объявления", value=False)
        # st.button("🧹 Очистить БД", type="primary", use_container_width=False, on_click=clear_db)
    
    with col2:
        
        if not scrape_all:
            num_pages = st.number_input("Количество страниц", min_value=1, max_value=100, value=1)
        else:
            num_pages = None
            st.info("Будут собраны все доступные объявления")
        st.button("🧹 Очистить БД", type="primary", use_container_width=False, on_click=clear_db)
    
    # Дополнительные параметры
    with st.expander("Дополнительные параметры"):
        region = st.selectbox("Регион", ["minsk", "gomel", "vitebsk", "grodno", "mogilev", "brest"], index=0)
        category = st.text_input("Категория", value="klavishnye")
    
    # Кнопка запуска
    if st.button("🚀 Начать парсинг", type="primary", use_container_width=True):
        if not scrape_all and num_pages is None:
            st.error("Выберите количество страниц или включите опцию 'Собрать все объявления'")
        else:
            with st.spinner("Парсинг в процессе..."):
                scraper = KufarScraper(delay=1.0, timeout=10)
                db = Database("keyscout.db")
                
                # Параметры поиска
                test_params = {
                    'mkb': 'v.or:1,25',
                    'mki': 'v.or:1,5'
                }
                
                try:
                    if scrape_all:
                        listings = scrape_all_pages(
                            scraper, 
                            region=region, 
                            category=category, 
                            **test_params
                        )
                    else:
                        listings = scraper.scrape_search_results(
                            region=region,
                            category=category,
                            max_pages=num_pages,
                            **test_params
                        )
                    
                    # Сохранение в БД
                    saved_count = db.save_listings(listings)
                    
                    st.success(f"✅ Парсинг завершен!")
                    st.info(f"📊 Найдено объявлений: {len(listings)}\n💾 Сохранено в БД: {saved_count}")
                    
                    # Сохраняем информацию о последнем парсинге в session state
                    st.session_state['last_scrape_count'] = len(listings)
                    st.session_state['last_scrape_saved'] = saved_count
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при парсинге: {str(e)}")
                finally:
                    db.close()
    
    # Показываем статистику последнего парсинга, если есть
    if 'last_scrape_count' in st.session_state:
        st.markdown("---")
        st.subheader("Последний парсинг")
        st.metric("Найдено объявлений", st.session_state['last_scrape_count'])
        st.metric("Сохранено в БД", st.session_state['last_scrape_saved'])

with tab2:
    st.title("📊 Результаты парсинга")
    st.markdown("---")
    
    # Загрузка данных из БД
    try:
        listings = get_listings_from_db()
        
        if not listings:
            st.info("📭 Объявлений пока нет. Запустите парсинг на странице настроек.")
        else:
            st.success(f"✅ Найдено объявлений: {len(listings)}")
            
            # Фильтры
            st.subheader("Фильтры")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_price_min = st.number_input("Минимальная цена", min_value=0, value=0)
            
            with col2:
                filter_price_max = st.number_input("Максимальная цена", min_value=0, value=0, 
                                                   help="0 = без ограничения")
            
            with col3:
                sort_by = st.selectbox("Сортировка", 
                                      ["Дата обновления (новые)", "Дата публикации", "Цена (по возрастанию)", "Цена (по убыванию)"])
            
            # Применение фильтров
            filtered_listings = listings.copy()
            
            if filter_price_min > 0:
                filtered_listings = [l for l in filtered_listings if l['price'] and l['price'] >= filter_price_min]
            
            if filter_price_max > 0:
                filtered_listings = [l for l in filtered_listings if l['price'] and l['price'] <= filter_price_max]
            
            # Сортировка
            if sort_by == "Дата обновления (новые)":
                filtered_listings.sort(key=lambda x: x['updated_at'] or '', reverse=True)
            elif sort_by == "Дата публикации":
                filtered_listings.sort(key=lambda x: x['published_at'] or datetime.min, reverse=True)
            elif sort_by == "Цена (по возрастанию)":
                filtered_listings.sort(key=lambda x: x['price'] or float('inf'))
            elif sort_by == "Цена (по убыванию)":
                filtered_listings.sort(key=lambda x: x['price'] or 0, reverse=True)
            
            st.markdown(f"**Показано объявлений: {len(filtered_listings)} из {len(listings)}**")
            st.markdown("---")
            
            # Отображение карточек
            for listing in filtered_listings:
                display_listing_card(listing)
                
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке данных: {str(e)}")
        st.exception(e)

