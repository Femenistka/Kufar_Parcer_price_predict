# Kufar Parser + Price Predict (Streamlit)

Проект собирает актуальные объявления с Kufar (б/у синтезаторы и цифровые пианино), извлекает из названия **бренд / модель / индекс модели**, обогащает объявления характеристиками из справочника и считает **оценку рыночной цены** с помощью ML-модели. Результаты сохраняются в SQLite и показываются в Streamlit UI (результаты, аналитика, “возможная выгода”).

## Интерфейс (Streamlit)

<details>
  <summary><b>1) Настройки парсинга</b></summary>
  <br/>
  <img width="900" alt="Настройки парсинга" src="https://github.com/user-attachments/assets/09c58fdd-eff6-47ad-9a33-f479881abb59" />
</details>

<details>
  <summary><b>2) Результаты (карточки + фильтры)</b></summary>
  <br/>
  <img width="900" alt="Результаты" src="https://github.com/user-attachments/assets/6a59d00c-4100-46eb-895c-e79153b36e6b" />
</details>

<details>
  <summary><b>3) Аналитика рынка (метрики)</b></summary>
  <br/>
  <img width="900" alt="Аналитика метрики" src="https://github.com/user-attachments/assets/9377c12a-04f9-413e-a115-8b5571a212ac" />
</details>

<details>
  <summary><b>4) Аналитика рынка (выгодные)</b></summary>
  <br/>
  <img width="900" alt="Аналитика выгодные" src="https://github.com/user-attachments/assets/54eea4ba-bf6f-4481-8185-3883946cb824" />
</details>



## Основные возможности
- Парсинг объявлений Kufar по выбранному региону и категории
- Извлечение из названия:
  - `Name` — бренд
  - `SubName` — модель
  - `IndexModel` — индекс модели (например P-45, P-55, P-95)
- Обогащение характеристиками из `Files/Характеристики_по_моделям.csv`
  - совпадение по (Brand, Model) — строго
  - если индекс модели не найден — используется **ближайший индекс** из справочника  
    Пример: объявление `Yamaha P 55`, а в справочнике есть только `Yamaha P 45` и `Yamaha P 95` → берём характеристики для `P 45`
- Расчёт `market_price` (рыночная оценка) по обученной модели `joblib`
- UI на Streamlit:
  - Настройки парсинга
  - Результаты (карточки + фильтры)
  - Аналитика рынка (метрики price vs market_price)
  - “Возможная выгода” (фильтр по диапазону ±% от рыночной оценки)

## Структура проекта
- `app.py` — Streamlit UI
- `scraper.py` — парсер Kufar + нормализация названий + работа с БД/обогащением
- `DB_functions.py` — вспомогательные функции для БД (например очистка)
- `models/` — ML-модели и метаданные
  - `models/SubName/price_model_precision.joblib`
  - `models/SubName+OTHERS/price_model_market.joblib`
- `Files/Характеристики_по_моделям.csv` — справочник характеристик инструментов
- `keyscout.db` — SQLite база (создаётся автоматически)

## Установка и запуск

### 1) Установить зависимости
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

<img width="1270" height="1033" alt="Схема работы парсера drawio" src="https://github.com/user-attachments/assets/e0dfb792-e20f-45b6-b9d6-21f4708e28dd" />


