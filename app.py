"""
Streamlit UI для парсера Kufar.
"""

import streamlit as st
import sqlite3
from datetime import datetime
from typing import List
from scraper import KufarScraper, Database, ListingRaw
import time
import pandas as pd
import numpy as np
from DB_functions import clear_db


# Настройка страницы
st.set_page_config(
    page_title="KeyScout - Парсер Kufar",
    page_icon="🎹",
    layout="wide"
)

def filter_possible_bargains(listings: list[dict], pct: float = 0.10) -> list[dict]:
    """
    Отбор "возможно выгодных" объявлений:
    price попадает в диапазон ±pct от market_price.
    Возвращает список dict, отсортированный по delta (price - market_price) по возрастанию
    (сначала те, что чуть дешевле рынка).
    """
    out = []

    for it in listings:
        price = it.get("price")
        mp = it.get("market_price")

        # нужны обе цены
        if price is None or mp is None:
            continue

        try:
            price = float(price)
            mp = float(mp)
        except (TypeError, ValueError):
            continue

        if mp <= 0:
            continue

        low = mp * (1 - pct)
        high = mp * (1 + pct)

        if low <= price <= high:
            it2 = dict(it)
            it2["delta"] = price - mp            # <0 выгоднее
            it2["delta_pct"] = (it2["delta"] / mp) * 100
            out.append(it2)

    out.sort(key=lambda x: x.get("delta", 0))
    return out


def build_market_analytics_df(listings: list[dict]) -> pd.DataFrame:
    """
    Безопасно строит DataFrame для аналитики.
    Никогда не падает, даже если БД пустая или таблицы только что очищены.
    """
    if not listings:
        return pd.DataFrame()

    df = pd.DataFrame(listings)

    required_cols = {"price", "market_price"}

    # если нужных колонок нет — возвращаем пустой df
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    # приводим к числам
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce")

    # оставляем только строки, где есть обе цены
    df = df.dropna(subset=["price", "market_price"]).copy()

    if df.empty:
        return df

    # вычисляем метрики
    df["delta"] = df["price"] - df["market_price"]
    df["gain"] = df["market_price"] - df["price"]
    df["gain_pct"] = (df["gain"] / df["market_price"]) * 100

    return df


def market_metrics(df: pd.DataFrame) -> dict:
    """Считает основные метрики рынка."""
    if df.empty:
        return {}

    metrics = {
        "Оценено объявлений (есть price + market_price)": int(len(df)),
        "Средняя разница price - market (Bias), BYN": float(df["delta"].mean()),
        "Медианная разница price - market, BYN": float(df["delta"].median()),
        "Среднее |price - market| (MAE), BYN": float(df["delta"].abs().mean()),
        "Медиана |price - market|, BYN": float(df["delta"].abs().median()),
        "Доля объявлений ниже рынка (price < market), %": float((df["price"] < df["market_price"]).mean() * 100),
        "Доля объявлений выше рынка (price > market), %": float((df["price"] > df["market_price"]).mean() * 100),
        "Макс переплата (price - market), BYN": float(df["delta"].max()),
        "Макс выгода (market - price), BYN": float(df["gain"].max()),
    }
    return metrics


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


def display_listing_card(listing_data: dict, add_benefits: bool = False):
    title = listing_data.get("title") or "Без названия"
    description = listing_data.get("description") or ""
    mp = listing_data.get("market_price")
    price = listing_data.get("price")
    currency = listing_data.get("currency") or "BYN"
    location = listing_data.get("location") or "Не указано"
    published_at = listing_data.get("published_at")
    url = listing_data.get("url")

    st.markdown(f"### {title}")
    if description:
        short = description[:200] + "..." if len(description) > 200 else description
        st.markdown(f"*{short}*")

    if add_benefits:
        left, right = st.columns(2)

        with left:
            st.caption("Цена в объявлении")
            st.markdown(f"## {format_price(price, currency) if price is not None else '—'}")

        with right:
            st.caption("Рыночная цена (модель)")
            st.markdown(f"## {format_price(mp, currency) if mp is not None else '💩 не хватило данных'}")

        if mp is not None and price is not None:
            diff = mp - price
            sign = "+" if diff > 0 else ""
            if diff > 0:
                st.success(f"Выгода: {sign}{format_price(diff, currency)}")
            else:
                st.warning(f"Переплата: {sign}{format_price(diff, currency)}")
    else:
        st.caption("Цена в объявлении")
        st.markdown(f"## {format_price(price, currency) if price is not None else '—'}")

    meta1, meta2 = st.columns(2)
    with meta1:
        st.caption("📍 Локация")
        st.write(location)
    with meta2:
        st.caption("📅 Дата")
        st.write(format_date(published_at))

    if url:
        st.markdown(f"[🔗 Открыть объявление]({url})")

    st.divider()


def get_listings_from_db(db_path: str = "keyscout.db") -> List[dict]:
    """Получение всех объявлений из БД."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT source_id, url, title, price, currency, published_at, 
               location, description, raw_text, created_at, updated_at, market_price
        FROM listings_enriched
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
            'updated_at': row['updated_at'],
            'market_price': row['market_price'],
        })
    
    return listings


# Главное меню - вкладки в шапке
tab1, tab2, tab3, tab4 = st.tabs(["Настройки парсинга", "Результаты", "Аналитика", "Возможная выгода"])

with tab1:
    st.title("🎹 KeyScout - Парсер объявлений Kufar")
    st.markdown("---")
    
    st.subheader("Настройки парсинга")
    
    # Параметры парсинга
    col1, col2 = st.columns(2)
    
    with col1:
        scrape_all = st.checkbox("Собрать все объявления", value=True)
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
                    

                    db = Database("keyscout.db")
                    n = db.load_model_specs_csv("Files/Характеристики_по_моделям.csv")  # путь свой
                    print("Характеристики_по_моделям загружены:", n)
                    # db.close()

                    # Сохранение в БД
                    saved_count = db.save_listings(listings)
                    # 1) парсинг -> listings
                    saved_ids = db.save_listings_return_ids(listings)
                    saved_count = len(saved_ids)

                    # 2) нормализация title -> Name/SubName/IndexModel
                    normalized_count = db.normalize_titles_for_ids(saved_ids)
                    
                    # 3) join -> listings_enriched
                    enriched_count = db.build_enriched_listings_table()
                    
                    # 4) predict + write back
                    stats = db.run_scoring_and_save_predictions(
                        model_path="models/SubName+OTHERS/price_model_market.joblib",
                        current_year=2026,
                        subname_min_count=3
                    )


                    st.success(f"✅ Парсинг завершен!")
                    st.info(
                        f"📊 Найдено: {len(listings)}\n"
                        f"💾 Сохранено: {len(saved_ids)}\n"
                        f"🧼 Нормализовано: {normalized_count}\n"
                        f"🔗 Обогащено характеристиками: {enriched_count}"
                    )

                    
                    # Сохраняем информацию о последнем парсинге в session state
                    st.session_state['last_scrape_count'] = len(listings)
                    st.session_state['last_scrape_saved'] = saved_count
                    st.session_state['last_normalized_count'] = normalized_count
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
        st.metric("Нормализовано", st.session_state['last_normalized_count'])

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

with tab3:
    st.title("📊 Аналитика рынка")

    # listings — это то, что ты уже получаешь через get_listings_from_db()
    df_m = build_market_analytics_df(listings)

    if df_m.empty:
        st.warning("Нет данных для аналитики: нужны объявления, где заполнены price и market_price.")
    else:
        # 1) Основные метрики
        m = market_metrics(df_m)

        # красивый вывод метрик в 3 колонки
        c1, c2, c3 = st.columns(3)
        c1.metric("Оценено объявлений", m["Оценено объявлений (есть price + market_price)"])
        c2.metric("Доля ниже рынка", f"{m['Доля объявлений ниже рынка (price < market), %']:.1f}%")
        c3.metric("Средняя |Δ| (MAE)", f"{m['Среднее |price - market| (MAE), BYN']:.0f} BYN")

        c1, c2, c3 = st.columns(3)
        c1.metric("Bias (price - market)", f"{m['Средняя разница price - market (Bias), BYN']:.0f} BYN")
        c2.metric("Макс выгода", f"{m['Макс выгода (market - price), BYN']:.0f} BYN")
        c3.metric("Макс переплата", f"{m['Макс переплата (price - market), BYN']:.0f} BYN")

        with st.expander("Показать все метрики"):
            st.json({k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()})

        st.divider()
        
        df_bargains = df_m

        st.caption(f"Найдено выгодных по фильтрам: {len(df_bargains)}")

        # сортировка: самые выгодные сверху
        df_bargains = df_bargains.sort_values(["gain", "gain_pct"], ascending=[False, False])

        # делаем обратно список dict для display_listing_card
        bargain_listings = df_bargains.to_dict(orient="records")

        # 3) Отображение карточек только выгодных
        for listing in bargain_listings:
            display_listing_card(listing, add_benefits=True)

with tab4:
    st.subheader("Возможная выгода (±15% от рыночной оценки)")

    # listings — список dict из get_listings_from_db()
    possible = filter_possible_bargains(listings, pct=0.15)

    if not possible:
        st.info("Нет объявлений с заполненными price и market_price, попадающих в диапазон ±10%.")
    else:
        # небольшой summary
        st.caption(f"Найдено: {len(possible)}")

        # можно добавить быстрый фильтр "показывать только чуть дешевле рынка"
        only_below = st.checkbox("Показывать только дешевле рынка", value=False)
        if only_below:
            possible = [x for x in possible if x.get("delta", 0) < 0]

        # вывод карточек
        for listing in possible:
            display_listing_card(listing, add_benefits=True)
