# =======================================================================
# ШАГ 1: Импорт библиотек
# =======================================================================
import os
import json
import ee
import geemap
import gradio as gr
import html

# =======================================================================
# ШАГ 2: Аутентификация и инициализация GEE (Адаптировано для сервера)
# =======================================================================
print("\n--- Инициализация Google Earth Engine ---")
try:
    # Render создаст файл по пути /etc/secrets/google_credentials.json
    secret_file_path = '/etc/secrets/google_credentials.json'
    if os.path.exists(secret_file_path):
        print("Аутентификация через секретный файл Render...")
        # Инициализация напрямую из файла
        credentials = ee.ServiceAccountCredentials(None, key_file=secret_file_path)
        ee.Initialize(credentials=credentials)
        print("✅ Аутентификация через сервисный аккаунт GEE прошла успешно.")
    else:
        # Резервный вариант для локального запуска
        print("Секрет не найден, попытка локальной аутентификации...")
        ee.Authenticate()
        ee.Initialize(project='gen-lang-client-0605302377')
        print("✅ Локальная аутентификация и инициализация GEE прошли успешно.")

except Exception as e:
    print(f"🔴 Критическая ошибка на этапе инициализации GEE: {e}")

# =======================================================================
# ШАГ 3: Обучение модели-классификатора
# =======================================================================
gee_classifier = None
bands_for_training = ['B2', 'B3', 'B4', 'B8', 'NDVI']

def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def train_classifier():
    global gee_classifier
    if gee_classifier:
        print("✅ Классификатор уже обучен.")
        return

    print("⏳ Обучение классификатора GEE... Это может занять около минуты.")
    try:
        desert = ee.FeatureCollection('projects/gen-lang-client-0605302377/assets/kalmykia_desert_samples').map(lambda f: f.set('class', 0))
        solonchak = ee.FeatureCollection('projects/gen-lang-client-0605302377/assets/kalmykia_solonchak_samples').map(lambda f: f.set('class', 1))
        arid = ee.FeatureCollection('projects/gen-lang-client-0605302377/assets/kalmykia_arid_samples').map(lambda f: f.set('class', 2))
        greenery = ee.FeatureCollection('projects/gen-lang-client-0605302377/assets/kalmykia_greenery_samples').map(lambda f: f.set('class', 3))
        water = ee.FeatureCollection('projects/gen-lang-client-0605302377/assets/kalmykia_water_samples').map(lambda f: f.set('class', 4))
        
        training_points = desert.merge(solonchak).merge(arid).merge(greenery).merge(water)
        roi_for_training = training_points.geometry().bounds()

        image_for_training = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate('2023-06-01', '2023-09-01') \
            .filterBounds(roi_for_training) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .map(add_ndvi) \
            .median() \
            .clip(roi_for_training)

        training_data = image_for_training.select(bands_for_training).sampleRegions(
            collection=training_points, properties=['class'], scale=10, tileScale=4)

        gee_classifier = ee.Classifier.smileRandomForest(numberOfTrees=50).train(
            features=training_data, classProperty='class', inputProperties=bands_for_training)
        print("✅ Модель-классификатор успешно обучена на 5 классах!")
    except Exception as e:
        print(f"🔴 Критическая ошибка во время обучения модели: {e}")

train_classifier()

# =======================================================================
# ШАГ 4: Функции для работы Gradio-приложения
# =======================================================================

def get_region_info(region_name):
    """Возвращает координаты и BBOX для выбранного региона."""
    regions = {
        "Элиста (тестовая область)": {
            "center": [46.307, 44.258],
            "bbox": [44.15, 46.25, 44.35, 46.35]
        },
        "Озеро Сарпа (Калмыкия)": {
            "center": [47.85, 44.75],
            "bbox": [44.65, 47.80, 44.85, 47.90]
        },
        "Озеро Нурын-Хаг (Калмыкия)": {
            "center": [46.817790, 45.354546],
            "bbox": [45.25, 46.71, 45.45, 46.91]
        },
        "Арзгир (Ставроп. край)": {
            "center": [45.38, 44.22],
            "bbox": [44.12, 45.33, 44.32, 45.43]
        },
        "Будённовск (Ставроп. край)": {
            "center": [44.78, 44.15],
            "bbox": [44.05, 44.73, 44.25, 44.83]
        },
        "Дельта Волги (Астрахань)": {
            "center": [46.0, 48.5],
            "bbox": [47.9, 45.6, 49.1, 46.4]
        },
        "Озеро Байкал (о. Ольхон)": {
            "center": [53.15, 107.34],
            "bbox": [106.84, 52.75, 107.84, 53.55]
        },
        "Москва (тестовая область)": {
            "center": [55.75, 37.61],
            "bbox": [37.5, 55.7, 37.7, 55.8]
        }
    }
    return regions.get(region_name)

def generate_classified_map(region_info, year, classifier):
    if not classifier: return None, "Ошибка: Классификатор не обучен."
    print(f"-- Запрос на генерацию карты для {year} года...")
    roi_to_classify = ee.Geometry.Rectangle(region_info['bbox'])
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(f'{year}-06-01', f'{year}-09-01') \
        .filterBounds(roi_to_classify) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25))
    if collection.size().getInfo() == 0: return None, f"⚠️ Не найдено снимков за {year} год."
    image_to_classify = collection.map(add_ndvi).median().clip(roi_to_classify)
    classified_image = image_to_classify.classify(classifier)
    class_palette = ['#e3a25a', '#ffffff', '#ffff00', '#00ff00', '#0000FF'] # Пустыня, Солончак, Сухая степь, Зелень, Вода
    vis_params = {'min': 0, 'max': 4, 'palette': class_palette}
    map_id = classified_image.getMapId(vis_params)
    print(f"-- ✅ Карта для {year} года сгенерирована.")
    return {"center": region_info['center'], "tile_url": map_id['tile_fetcher'].url_format, "year": year}, None

def create_map_iframe_html(map_data, region_name):
    center, tile_url, year = map_data['center'], map_data['tile_url'], map_data['year']
    iframe_content = f"""
    <!DOCTYPE html><html><head><title>Карта {year}</title><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0"><link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" /><script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script><style>html, body, #map {{ height: 100%; width: 100%; margin: 0; padding: 0; }}</style></head><body><div id="map"></div><script>
        var map = L.map('map').setView([{center[0]}, {center[1]}], 12);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ attribution: '&copy; OpenStreetMap' }}).addTo(map);
        L.tileLayer(`{tile_url}`, {{ attribution: 'Google Earth Engine' }}).addTo(map);
    </script></body></html>"""
    escaped_html = html.escape(iframe_content)
    return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 500px; border: 1px solid #ccc;"></iframe>'

def process_and_display_maps(region_name, year1, year2, year3):
    """Основная функция, вызываемая Gradio."""
    if not gee_classifier:
        # ИСПРАВЛЕНО: Возвращаем 4 значения, чтобы избежать ошибки
        return None, None, None, "Ошибка: модель не обучена. Проверьте логи на наличие проблем с GEE."

    region_info = get_region_info(region_name)
   
    years = sorted(list(set([year1, year2, year3]))) #было reverse=True
    
    outputs, messages = [], []
    print(f"\n🚀 Новый запрос! Регион: {region_name}, Годы: {years}")
    for year in years:
        map_data, msg = generate_classified_map(region_info, year, gee_classifier)
        if map_data: outputs.append(create_map_iframe_html(map_data, region_name))
        else: outputs.append(f"<p style='text-align:center; padding-top: 200px;'>{msg}</p>")
        if msg: messages.append(msg)
    
    while len(outputs) < 3: outputs.append(None)
    
    final_message = "✅ Готово. " + " ".join(messages)
    return outputs[0], outputs[1], outputs[2], final_message


# =======================================================================
# ШАГ 5: Создание и запуск интерфейса Gradio
# =======================================================================
with gr.Blocks(
    css="""
    /*
       Этот CSS использует самый надежный метод:
       1. Фон применяется ко всей странице (body).
       2. Используется прямая, абсолютная ссылка на вашу картинку на GitHub.
       3. !important используется, чтобы перебить любые стандартные стили Gradio.
       4. Контейнеры приложения делаются прозрачными, чтобы фон был виден.
    */
    body, gradio-app {
        /* Используем вашу точную ссылку и добавляем !important для надежности */
        background-image: url("https://raw.githubusercontent.com/AnnaZverev/Landcover/refs/heads/main/picture.jpg") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        min-height: 100vh;
    }

    /* Убираем ограничение по ширине и делаем основной контейнер прозрачным */
    .gradio-container {
        max-width: none !important;
        background: transparent !important;
    }

    /* Делаем все дочерние блоки тоже прозрачными, чтобы фон был виден везде */
    .gradio-container .gr-panel, .gradio-container .gr-form, .gradio-container .gr-row {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Добавляем красивую полупрозрачную "подложку" только для левой колонки с элементами управления */
    .gradio-container .gr-column {
        background-color: rgba(0, 0, 0, 0.5) !important; /* Черный, на 50% прозрачный */
        border-radius: 15px !important;
        padding: 20px !important;
    }


    /* Сохраняем стили для текста, чтобы он был читаемым */
    .gradio-container h1, .gradio-container p, .gradio-container label, .gradio-container .message, .gradio-container .gr-button-primary {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    """
) as demo:

    _ = gr.Image('picture.jpg', visible=False, interactive=False)
    gr.Markdown("# 🛰️ Анализ почвенного покрова")
    gr.Markdown("Выберите регион и до трёх лет для анализа. Карты будут будут показаны слева направо (от более ранних годов к более поздним).")
    with gr.Row():
        with gr.Column(scale=1):
            region_dropdown = gr.Dropdown(
                label="Регион",
                choices=[
                    "Элиста (тестовая область)",
                    "Озеро Сарпа (Калмыкия)",
                    "Озеро Нурын-Хаг (Калмыкия)",
                    "Арзгир (Ставроп. край)",
                    "Будённовск (Ставроп. край)",
                    "Дельта Волги (Астрахань)",
                    "Озеро Байкал (о. Ольхон)",
                    "Москва (тестовая область)"
                ],
                value="Элиста (тестовая область)")
            year1_slider = gr.Slider(label="Год 1", minimum=2019, maximum=2025, step=1, value=2019)
            year2_slider = gr.Slider(label="Год 2", minimum=2019, maximum=2025, step=1, value=2021)
            year3_slider = gr.Slider(label="Год 3", minimum=2019, maximum=2025, step=1, value=2023)
            submit_button = gr.Button("Сгенерировать карты", variant="primary")
            status_message = gr.Markdown()
    with gr.Row():
        map1_output = gr.HTML()
        map2_output = gr.HTML()
        map3_output = gr.HTML()
    submit_button.click(
        fn=process_and_display_maps,
        inputs=[region_dropdown, year1_slider, year2_slider, year3_slider],
        outputs=[map1_output, map2_output, map3_output, status_message])

# --- ЗАПУСК ПРИЛОЖЕНИЯ ---
print("\n--- Запуск Gradio интерфейса ---")
# Получаем порт из переменной окружения PORT, которую предоставляет Render.
# Если ее нет (при локальном запуске), используем 7860 по умолчанию.
port = int(os.environ.get('PORT', 7860))
# Запускаем сервер, чтобы он был доступен извне контейнера
demo.launch(server_name="0.0.0.0", server_port=port)











