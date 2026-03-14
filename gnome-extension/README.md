# GNOME On-Screen Translator (OptikR backend)

This folder contains a GNOME Shell extension plus a local Python backend runner.

## What was added

- Extension path:
  - `gnome-extension/on-screen-translator@optikr/metadata.json`
  - `gnome-extension/on-screen-translator@optikr/extension.js`
- Backend runner:
  - `clean/run_gnome_backend.py`

## Backend setup (Linux machine)

Run from the `clean` directory:

```bash
python run_gnome_backend.py --host 127.0.0.1 --port 8765
```

Then verify:

```bash
curl http://127.0.0.1:8765/health
```

Expected response includes discovered OCR and translation plugins.

## Extension install

Copy extension folder to your GNOME extensions directory:

```bash
mkdir -p ~/.local/share/gnome-shell/extensions
cp -r "on-screen-translator@optikr" ~/.local/share/gnome-shell/extensions/
```

Reload GNOME Shell and enable:

```bash
gnome-extensions enable on-screen-translator@optikr
```

## Usage

1. Start backend (`run_gnome_backend.py`).
2. Click the panel dictionary icon.
3. Use **Capture Area and Translate**.
4. Select screen region.
5. Result appears as a GNOME notification and is copied to clipboard.

## Notes

- Current flow uses `gnome-screenshot -a` for area selection.
- Default engines are `tesseract` (OCR) and `google_free` (translation).
- This is an MVP extension path; if you want, next step is native overlay text
  rendering (instead of notification-only) and keyboard shortcut settings.
