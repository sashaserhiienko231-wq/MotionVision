# Motion Vision 1.0.0 — отчет о готовности релиза

Дата проверки: **29 сентября 2026**. Локальный runner: Windows x64 (`Windows 10.0.26200`). Hosted matrix: GitHub Actions run [36542162073](https://github.com/sashaserhiienko231-wq/MotionVision/actions/runs/36542162073), tested code commit `1635c0b7566025f5032e77057b3895058d611153`.

## Статус платформ

| Платформа | Hosted runner / фактическая архитектура | GitHub Actions artifact (uploaded ZIP size) | Реальные файлы в artifact | Hosted tests |
|---|---|---|---|---|
| Windows x64 | `windows-latest` / x64 | [MotionVision-v1.0.0-Windows-x64](https://github.com/sashaserhiienko231-wq/MotionVision/actions/runs/36542162073/artifacts/11020629443) — **304 057 465 байт** | EXE **152 260 767 байт**; installer **153 182 997 байт** | 78/78 Python tests passed |
| Linux x86_64 | `ubuntu-latest` / Ubuntu 24.04, x86_64 | [MotionVision-v1.0.0-Linux-x86_64](https://github.com/sashaserhiienko231-wq/MotionVision/actions/runs/36542162073/artifacts/11020484870) — **386 113 393 байт** | AppImage **193 307 840 байт**; DEB **192 850 538 байт** | 78/78 Python tests passed |
| macOS ARM64 | `macos-latest` / arm64 | [MotionVision-v1.0.0-macOS-arm64](https://github.com/sashaserhiienko231-wq/MotionVision/actions/runs/36542162073/artifacts/11020729053) — **521 891 574 байта** | `Motion Vision.app` ~304 MiB; DMG ~156 MiB | 78/78 Python tests passed |
| Android multiarch | `ubuntu-latest` / Ubuntu 24.04, amd64, JDK 17 | [MotionVision-v1.0.0-Android-multiarch](https://github.com/sashaserhiienko231-wq/MotionVision/actions/runs/36542162073/artifacts/11020694685) — **81 252 995 байт** | APK **78 383 585 байт**; AAB **40 516 424 байта** | 78/78 Python tests + 4/4 Gradle unit tests passed |

**GitHub Actions run 36542162073: все четыре jobs завершились успешно и загрузили artifacts.** Windows EXE icon, Linux DEB/AppImage icon, macOS `.icns` и plist metadata, Android signing/package ID/launcher icon checks прошли. Linux AppImage/DEB и macOS app/DMG собраны реальными native runners, не локальными заглушками.

Размер в таблице — размер загруженного GitHub Actions artifact ZIP из GitHub API; размеры отдельных файлов приведены из native job logs. Artifacts настроены с хранением **30 дней**.

Каждый hosted job запускает 78 общих Python tests. Android job дополнительно запускает 4 Kotlin/Gradle unit tests. Всего в этом run: 312 Python test executions и 4 Android unit test executions.

## Локальные артефакты

| Файл | Размер |
|---|---:|
| `dist/windows/MotionVision.exe` | 152 259 294 байта |
| `dist/windows/MotionVision-Setup.exe` | 152 905 522 байта |
| `dist/android/MotionVision.apk` | 78 383 585 байт |
| `dist/android/MotionVision.aab` | 40 516 424 байт |
| Linux AppImage / DEB | Не создавались локально; собраны и загружены hosted job |
| macOS `.app` / DMG | Не создавались локально; собраны и загружены hosted job |

Android AAB подписан debug-ключом и не предназначен для загрузки в Play Store. Windows installer и desktop-приложение не подписаны кодовым сертификатом.

## Известные ограничения

- Ручной workflow `.github/workflows/release.yml` запускается через `workflow_dispatch`; он создает build artifacts, но не публикует GitHub Release.
- macOS runner сейчас arm64; отдельный Intel runner не настроен. macOS app/DMG не подписаны и не notarized; Gatekeeper может потребовать разрешение пользователя.
- Linux и macOS файлы проверялись на hosted runners, но приложение не запускалось на локальных Linux/macOS устройствах с камерой.
- Android APK/AAB собраны и подписаны debug-ключом; приложение не было запущено на физическом Android устройстве или эмуляторе.
- Windows installer и EXE не подписаны кодовым сертификатом. Для production Android AAB нужен пользовательский release keystore.
- Telegram проверен только mock-тестами без credentials. Desktop face embeddings локальные, но приложение их не шифрует.

## Автоматические проверки

- Python desktop suite: **78/78 тестов пройдены**. Включает геометрию жестов и их временную стабилизацию, две руки, несколько лиц и стабильные ID, SFace matching и отключение face recognition, fallback камеры, запись, mock Telegram, window state, fullscreen и HUD layout.
- Android `testDebugUnitTest`: **4/4 теста пройдены**.
- `compileall` для desktop core/tests: пройдено.
- `pip check` project venv: `No broken requirements found`.
- Windows hosted one-file build, запуск packaged executable с `--version`, встроенные version/icon ресурсы и компиляция Inno Setup: пройдены.
- Android `assembleRelease`, `bundleRelease` и `lintVitalRelease`: пройдены. В APK включены ABI `arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`.
- Linux hosted AppImage и DEB созданы; DEB metadata, icon in DEB, AppImage extraction and bundled icon проверены.
- macOS hosted `Motion Vision.app` и DMG созданы; `plutil -lint`, CFBundle metadata/version fields, `.icns` resource and DMG checksum прошли.
- `apksigner verify` для APK: подпись v2 проверена; `jarsigner -verify` для AAB: `jar verified`. Подпись Android — локальная debug, self-signed.
- Android manifest: запрашивается камера; INTERNET и ACCESS_NETWORK_STATE не включены.
- `bash -n` прошел для Linux, macOS, Android и Linux aggregate scripts.
- Оба workflow прошли локальный YAML parse; каждый hosted job завершился successfully и загрузил названный artifact.

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
