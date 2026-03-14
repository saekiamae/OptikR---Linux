import Gio from "gi://Gio";
import GLib from "gi://GLib";
import Soup from "gi://Soup?version=3.0";
import St from "gi://St";
import * as Main from "resource:///org/gnome/shell/ui/main.js";
import * as PanelMenu from "resource:///org/gnome/shell/ui/panelMenu.js";
import * as PopupMenu from "resource:///org/gnome/shell/ui/popupMenu.js";
import { Extension } from "resource:///org/gnome/shell/extensions/extension.js";

const BACKEND_URL = "http://127.0.0.1:8765/ocr-translate";

class TranslatorButton extends PanelMenu.Button {
    constructor() {
        super(0.0, "On-Screen Translator");

        const icon = new St.Icon({
            icon_name: "accessories-dictionary-symbolic",
            style_class: "system-status-icon",
        });
        this.add_child(icon);

        this._http = new Soup.Session();

        this._addMenuItems();
    }

    _addMenuItems() {
        const captureItem = new PopupMenu.PopupMenuItem("Capture Area and Translate");
        captureItem.connect("activate", () => {
            void this._captureAndTranslateArea();
        });
        this.menu.addMenuItem(captureItem);

        const testItem = new PopupMenu.PopupMenuItem("Check Backend Health");
        testItem.connect("activate", () => {
            void this._checkBackend();
        });
        this.menu.addMenuItem(testItem);
    }

    async _checkBackend() {
        try {
            const msg = Soup.Message.new("GET", "http://127.0.0.1:8765/health");
            const bytes = await this._http.send_and_read_async(msg, GLib.PRIORITY_DEFAULT, null);
            const body = new TextDecoder().decode(bytes.get_data());
            const payload = JSON.parse(body);
            Main.notify("On-Screen Translator", `Backend OK. OCR plugins: ${payload.ocr_plugins.length}`);
        } catch (error) {
            Main.notify("On-Screen Translator", `Backend check failed: ${error}`);
        }
    }

    _runScreenshotArea(outputPath) {
        const shellCmd = `gnome-screenshot -a -f "${outputPath}"`;
        try {
            GLib.spawn_command_line_sync(shellCmd);
        } catch (error) {
            throw new Error(`Screenshot command failed: ${error}`);
        }
    }

    async _captureAndTranslateArea() {
        const tempPath = `/tmp/optikr_capture_${Date.now()}.png`;
        Main.notify("On-Screen Translator", "Select a region to translate...");

        try {
            this._runScreenshotArea(tempPath);
            if (!GLib.file_test(tempPath, GLib.FileTest.EXISTS)) {
                Main.notify("On-Screen Translator", "Capture canceled.");
                return;
            }

            const payload = {
                image_path: tempPath,
                source_lang: "auto",
                target_lang: "en",
                ocr_engine: "tesseract",
                translation_engine: "google_free",
                min_confidence: 0.25,
            };

            const message = Soup.Message.new("POST", BACKEND_URL);
            message.set_request_body_from_bytes(
                "application/json",
                new GLib.Bytes(new TextEncoder().encode(JSON.stringify(payload)))
            );

            const bytes = await this._http.send_and_read_async(message, GLib.PRIORITY_DEFAULT, null);
            const body = new TextDecoder().decode(bytes.get_data());
            const result = JSON.parse(body);

            if (!result.success) {
                Main.notify("On-Screen Translator", `Backend error: ${result.error ?? "unknown"}`);
                return;
            }

            if (!result.translated_text) {
                Main.notify("On-Screen Translator", "No text detected.");
                return;
            }

            const translated = result.translated_text;
            const preview = translated.length > 220 ? `${translated.slice(0, 220)}...` : translated;
            Main.notify("On-Screen Translator", preview);

            const clipboard = St.Clipboard.get_default();
            clipboard.set_text(St.ClipboardType.CLIPBOARD, translated);
        } catch (error) {
            Main.notify("On-Screen Translator", `Translation failed: ${error}`);
        } finally {
            if (GLib.file_test(tempPath, GLib.FileTest.EXISTS)) {
                try {
                    Gio.File.new_for_path(tempPath).delete(null);
                } catch (_) {
                    // best effort cleanup
                }
            }
        }
    }
}

export default class OnScreenTranslatorExtension extends Extension {
    enable() {
        this._button = new TranslatorButton();
        Main.panel.addToStatusArea("on-screen-translator-optikr", this._button);
    }

    disable() {
        if (this._button) {
            this._button.destroy();
            this._button = null;
        }
    }
}
