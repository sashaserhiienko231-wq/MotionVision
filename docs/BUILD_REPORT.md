# Motion Vision 1.0.0 — отчет о готовности релиза

Дата локальной проверки: **29 сентября 2026**. Локальный runner: Windows x64 (`Windows 10.0.26200`). Исходники прежнего приложения вне `MotionVision/` не менялись.

## Статус платформ

| Платформа | Runner, заданный для CI | Имя GitHub Actions artifact | Статус и фактический локальный результат | Тесты |
|---|---|---|---|---|
| Windows x64 | `windows-latest` | `MotionVision-v1.0.0-Windows-x64` | Локально собран и проверен: EXE 152 259 294 байта; installer 152 905 522 байта | 78 Python tests passed локально |
| Android multiarch | `ubuntu-latest` | `MotionVision-v1.0.0-Android-multiarch` | Локально собран и проверен на Windows runner: APK 78 383 585 байт; AAB 40 516 424 байта | 4 Gradle unit tests passed; 78 Python tests separately passed |
| Linux x86_64 | `ubuntu-latest` | `MotionVision-v1.0.0-Linux-x86_64` | **Не собран**; локального AppImage/DEB нет; ожидаются AppImage и DEB | Linux runner не запускался; 78 общих Python tests прошли только на Windows |
| macOS | `macos-latest` | `MotionVision-v1.0.0-macOS-<uname -m>` | **Не собран**; локального `.app`/DMG нет; ожидаются `Motion Vision.app` и DMG | macOS runner не запускался; 78 общих Python tests прошли только на Windows |

**GitHub Actions verification: ни один job пока не запускался.** Имена runners выше — конфигурация, а не свидетельство запуска. Windows и Android существуют как локально собранные артефакты; Linux и macOS должны оставаться непроверенными до успешных нативных jobs.

Локальный Git repository и ветка `main` созданы; исходники и workflows подготовлены в начальном локальном commit. Git remote не настроен, поэтому GitHub Actions еще не запускался. После создания GitHub repository нужен его фактический URL, например `https://github.com/<OWNER>/MotionVision.git`; владельца и адрес здесь не придумывали. Добавьте этот URL как `origin` и выполните `git push -u origin main`, чтобы запустить CI.

GitHub Actions выполняет общие 78 Python тестов в каждом desktop job, затем нативные проверки. Android job также запускает 4 Kotlin/Gradle unit tests. Будущие native результаты Linux/macOS будут отражены отдельно после реального запуска workflow.

## Локальные артефакты

| Файл | Размер |
|---|---:|
| `dist/windows/MotionVision.exe` | 152 259 294 байта |
| `dist/windows/MotionVision-Setup.exe` | 152 905 522 байта |
| `dist/android/MotionVision.apk` | 78 383 585 байт |
| `dist/android/MotionVision.aab` | 40 516 424 байт |
| Linux AppImage / DEB | Не созданы |
| macOS `.app` / DMG | Не созданы |

Android AAB подписан debug-ключом и не предназначен для загрузки в Play Store. Windows installer и desktop-приложение не подписаны кодовым сертификатом.

## Известные ограничения

- GitHub Actions пока не запускался, поэтому здесь нет CI-загрузок или нативной Linux/macOS проверки. Для ручного релиза подготовлен `.github/workflows/release.yml` с `workflow_dispatch`.
- Linux и macOS артефакты и их размеры появятся только после успешных hosted jobs. Текущие локальные Python test results не заменяют native Linux/macOS результаты.
- Android APK/AAB собраны и подписаны debug-ключом; приложение не было запущено на физическом Android устройстве или эмуляторе.
- Windows installer и macOS app/DMG остаются неподписанными/notarized. Для production Android AAB нужен пользовательский release keystore.
- Telegram проверен только mock-тестами без credentials. Desktop face embeddings локальные, но приложение их не шифрует.

## Автоматические проверки

- Python desktop suite: **78/78 тестов пройдены**. Включает геометрию жестов и их временную стабилизацию, две руки, несколько лиц и стабильные ID, SFace matching и отключение face recognition, fallback камеры, запись, mock Telegram, window state, fullscreen и HUD layout.
- Android `testDebugUnitTest`: **4/4 теста пройдены**.
- `compileall` для desktop core/tests: пройдено.
- `pip check` project venv: `No broken requirements found`.
- Windows one-file сборка, запуск packaged executable с `--version`, встроенные version/icon ресурсы и компиляция Inno Setup: пройдены.
- Android `assembleRelease`, `bundleRelease` и `lintVitalRelease`: пройдены. В APK включены ABI `arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`.
- `apksigner verify` для APK: подпись v2 проверена; `jarsigner -verify` для AAB: `jar verified`. Подпись Android — локальная debug, self-signed.
- Android manifest: запрашивается камера; INTERNET и ACCESS_NETWORK_STATE не включены.
- `bash -n` прошел для Linux, macOS, Android и Linux aggregate scripts.
- Оба workflow прошли локальный YAML parse; Windows PE icon resource и Android APK signing/package/icon checks прошли.

MediaPipe во время smoke-тестов печатает предупреждение о `NORM_RECT` без `IMAGE_DIMENSIONS`; smoke-тест при этом прошел. PyInstaller также выводит предупреждение об экспериментальном NumPy `array_api` и необязательных зависимостях MediaPipe GenAI конвертера; приложение собирается без этих необязательных пакетов.

## Проверка камеры и окна

В живом desktop прогоне с подключенной камерой обработано 53 кадра. Наблюдалось:

| Показатель | Измерение |
|---|---:|
| Фактическое разрешение | 640×480 |
| Camera FPS | 27.86 |
| Detection FPS | 46.81 |
| Render FPS | 28.52 |
| Inference time | 24.07 мс |
| Frame latency | 39.32 мс |

Запрашивались 30 FPS; 4K/120 FPS не заявляются. **Самый высокий режим, подтвержденный в этой сессии, — 640×480 примерно при 28 FPS.** Это измеренный режим данного прогона, а не предел возможностей камеры: другие разрешения на этой камере не подтверждались.

В живом desktop окне проверены minimize/restore/maximize и F11 с переходом на экран 1920×1080. Упакованное Windows окно запускалось, отображалось и проходило minimize/restore/maximize. Автоматизированная отправка F11 в packaged build не дала однозначного результата, поэтому fullscreen подтвержден на исходном desktop запуске, а не отдельно на EXE.

Камера Android не проверялась на устройстве или эмуляторе: подключенного Android target не было. Поэтому Android FPS, GPU delegate, смена камер и воспроизведение записанного файла не измерялись. Telegram-транспорт проверен только mock-тестами; действительные credentials не задавались, сообщение не отправлялось.

## Команды

Локальная установка desktop зависимостей и запуск из корня проекта в PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config\.env.example .env
python core\app.py
```

Готовый Windows build запускается через `dist\windows\MotionVision.exe`. Android APK устанавливается командой `adb install -r .\dist\android\MotionVision.apk`. Сборка Windows+Android: `.\scripts\build_all.ps1`; Linux+Android на Linux: `bash scripts/build_all.sh`. После успешного GitHub Linux job Debian пакет устанавливается командой `sudo apt install ./MotionVision.deb`; после macOS job DMG открывается командой `open MotionVision.dmg`.

Desktop `.env` поддерживает `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `AUTO_RECORD_ON_START`, `FACE_RECOGNITION_ENABLED`, `FACE_MATCH_THRESHOLD`, `FACE_ALERT_COOLDOWN`, `CAMERA_INDEX`, `CAMERA_RESOLUTION` и `CAMERA_FPS`. В репозитории хранится только пустой шаблон `config/.env.example`.

## Что реализовано по файлам

Ниже перечислены добавленные исходники и групповые ресурсные файлы нового корня. Локальные `.venv`, Gradle/PyInstaller промежуточные файлы и каталоги `dist/` являются генерируемыми; они не перечисляются как исходники.

### Корень, конфигурация и CI

- `.gitignore` исключает локальные секреты, runtime данные, виртуальное окружение, кэш и сборки.
- `.github/workflows/build.yml` задает native jobs на `windows-latest`, `ubuntu-latest`, `macos-latest` и reusable `workflow_call`.
- `.github/workflows/release.yml` вручную запускает ту же reusable build matrix через `workflow_dispatch`.
- `README.md` описывает структуру проекта, платформенное покрытие, запуск и сборку.
- `requirements.txt`, `requirements-build.txt` задают runtime- и build-зависимости Python.
- `config/version.json` — версия и Android package ID; `config/.env.example` — документированный шаблон настроек без токенов, включая флаг отключения face matching.
- `motion_vision_config.json` содержит стартовые локальные настройки камеры и режима обработки.
- `models/face_landmarker.task`, `models/hand_landmarker.task`, `models/pose_landmarker_lite.task` и `models/face_recognition_sface_2021dec.onnx` — реальные локальные модели. Повторные `.task.download` копии удалены.
- `MotionVision.spec` — генерируемый PyInstaller spec для Windows сборки.

### Desktop core

- `core/app.py` запускает desktop UI/pipeline, загружает модели, обрабатывает `--version` и пропускает embeddings/matching/enrollment при отключенном face recognition.
- `core/camera.py` управляет camera negotiation, наблюдаемыми режимами и восстановлением после ошибок.
- `core/actions.py` распознает жесты с геометрическими признаками, сглаживанием, confidence и cooldown.
- `core/app_config.py` читает настройки и переменные окружения, включая `FACE_RECOGNITION_ENABLED`.
- `core/face_identity.py` ведет локальные face embeddings, enrollment, matching и трекинг.
- `core/sface.py` оборачивает OpenCV SFace embedding inference.
- `core/telegram_bot.py` реализует desktop Telegram Bot API и обработку событий/записей.
- `core/hud_drag.py` управляет перемещением и сохранением положения HUD-панелей.
- `core/windowing.py` сохраняет и восстанавливает безопасную геометрию окна.
- `core/paths.py` задает пути к упакованным ресурсам и пользовательским данным; `core/version.py` читает метаданные версии.

### Android

- `android/settings.gradle.kts`, `android/build.gradle.kts`, `android/gradle.properties`, `android/gradle/wrapper/gradle-wrapper.properties`, `android/gradle/wrapper/gradle-wrapper.jar`, `android/gradlew`, `android/gradlew.bat` — Gradle project и закрепленный wrapper.
- `android/app/build.gradle.kts` настраивает CameraX, MediaPipe Tasks, SDK levels, ABI и signing; `android/app/proguard-rules.pro` содержит правила shrinker.
- `android/app/src/main/AndroidManifest.xml` задает package, разрешение камеры и activity.
- `MainActivity.kt` управляет нативным экраном, CameraX lifecycle, переключением камеры, fullscreen и записью; `VisionPipeline.kt` запускает MediaPipe Tasks и fallback CPU; `VisionModels.kt` содержит tracking/gesture модели; `VisionOverlayView.kt` рисует landmarks и overlay.
- `android/app/src/test/java/com/motionvision/app/VisionModelsTest.kt` проверяет трекинг и геометрические/временные свойства жестов.
- `android/app/src/main/assets/{face_landmarker.task,hand_landmarker.task,pose_landmarker_lite.task}` — модели Android.
- `android/app/src/main/res/values/{colors.xml,styles.xml}` — тема/цвета; `mipmap-anydpi-v26/ic_launcher.xml` и `mipmap-anydpi-v33/ic_launcher.xml` — adaptive/monochrome icon declarations.
- `android/app/src/main/res/mipmap-{mdpi,hdpi,xhdpi,xxhdpi,xxxhdpi}/` содержит обычные, round, foreground и monochrome launcher PNG icons.

### Иконки и платформенная упаковка

- `assets/icon/icon.svg`, `icon-foreground.svg`, `icon-monochrome.svg` — исходная векторная, Android foreground и monochrome версии.
- `assets/icon/png/motion-vision-{16,32,48,64,128,256,512,1024}.png` — заданные raster-размеры; `assets/icon/MotionVision.ico` и `MotionVision.icns` — Windows/macOS контейнеры.
- `assets/icon/MotionVision.iconset/icon_{16x16,16x16@2x,32x32,32x32@2x,128x128,128x128@2x,256x256,256x256@2x,512x512,512x512@2x}.png` — macOS iconset.
- `assets/platform/linux/hicolor/{16x16,24x24,32x32,48x48,64x64,128x128,256x256,512x512}/apps/com.motionvision.app.png` — Linux icon theme.
- `linux/MotionVision.desktop` — Linux desktop entry.
- `windows/MotionVision.iss`, `windows/version-info.txt` — Inno Setup installer и Windows file metadata.
- `scripts/build_windows.ps1`, `build_android.ps1`, `build_linux.sh`, `build_macos.sh` — нативные сборки; `build_all.ps1` и `build_all.sh` — агрегаторы; `generate_icons.py` воспроизводимо генерирует иконки; `verify_windows_icon.py` проверяет PE icon resources.
- `docs/ARCHITECTURE.md`, `docs/BUILD.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/BUILD_REPORT.md` описывают архитектуру, сборки, ограничения и результаты этого прогона.

### Тесты

- `tests/test_actions.py` — распознавание, debounce, confidence и smoothing жестов.
- `tests/test_app.py` — адаптивный HUD, рендер-геометрия, режимы обработки и window transitions.
- `tests/test_features.py` — камера, модели, несколько лиц, SFace, выключатель face recognition, запись и mock Telegram.
- `tests/test_hud_drag.py` — drag/pinch и сохранение панелей.
- `tests/test_windowing.py` — безопасная геометрия, monitor work area и восстановление.
