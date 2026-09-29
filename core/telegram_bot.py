"""Small Telegram Bot API client with a mockable transport and polling thread."""
from __future__ import annotations

import json
import mimetypes
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class TelegramError(RuntimeError):
    pass


@dataclass(frozen=True)
class BotCommand:
    chat_id: str
    command: str
    args: tuple[str, ...] = ()
    photo: bytes | None = None


class TelegramBot:
    def __init__(
        self,
        token: str,
        chat_id: str = "",
        *,
        transport: Any | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._token = token.strip()
        self.chat_id = str(chat_id).strip()
        self.transport = transport or TelegramTransport(self._token)
        self.on_status = on_status or (lambda _: None)
        self.commands: queue.SimpleQueue[BotCommand] = queue.SimpleQueue()
        self.awaiting_face: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self.connected = False

    @property
    def enabled(self) -> bool:
        return bool(self._token)

    def start(self) -> bool:
        if not self.enabled:
            self.on_status("Telegram is not configured; local recording remains available")
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="motion-vision-telegram", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                updates = self.transport.get_updates(self._offset, timeout=20)
                self.connected = True
                self.on_status("Telegram connected")
                for update in updates:
                    self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                    self.handle_update(update)
            except Exception as exc:
                self.connected = False
                # Do not include API URL/transport exception text; it may contain the token.
                self.on_status(f"Telegram unavailable ({type(exc).__name__}); retrying")
                self._stop.wait(3.0)

    def handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if not chat_id or chat.get("type", "private") != "private":
            return
        if self.chat_id and chat_id != self.chat_id:
            return
        if not self.chat_id:
            self.chat_id = chat_id
        text = str(message.get("text", "")).strip()
        if text.startswith("/"):
            parts = text.split()
            command = parts[0].split("@", 1)[0].lower()
            args = tuple(parts[1:])
            if command in {"/start", "/help"}:
                self.send_message(chat_id, "Motion Vision commands: /status /photo /record /stop /history /settings /addface /faces /deleteface ID /clearfaces /watch")
                return
            if command == "/addface":
                self.awaiting_face[chat_id] = " ".join(args).strip()
                self.send_message(chat_id, "Send one clear face photo. It will be processed locally and discarded after enrollment.")
                return
            if command == "/cancel":
                self.awaiting_face.pop(chat_id, None)
                self.send_message(chat_id, "Face enrollment cancelled.")
                return
            if command in {"/status", "/photo", "/record", "/stop", "/history", "/settings", "/faces", "/deleteface", "/clearfaces", "/watch"}:
                self.commands.put(BotCommand(chat_id, command, args))
                return
            self.send_message(chat_id, "Unknown command. Send /help for available commands.")
            return
        if chat_id in self.awaiting_face and message.get("photo"):
            name = self.awaiting_face.pop(chat_id)
            try:
                photo = message["photo"][-1]
                file_info = self.transport.get_file(photo["file_id"])
                image = self.transport.download_file(file_info["file_path"])
                self.commands.put(BotCommand(chat_id, "/addface_photo", (name,) if name else (), image))
            except Exception as exc:
                self.send_message(chat_id, f"Could not download photo ({type(exc).__name__}).")
        elif chat_id in self.awaiting_face:
            self.send_message(chat_id, "Please send a photo, or use /cancel to stop enrollment.")

    def send_message(self, chat_id: str, text: str) -> bool:
        try:
            self.transport.call("sendMessage", {"chat_id": str(chat_id), "text": text})
            return True
        except Exception as exc:
            self.connected = False
            self.on_status(f"Telegram send failed ({type(exc).__name__})")
            return False

    def send_file(self, chat_id: str, method: str, path: str, caption: str = "") -> bool:
        try:
            self.transport.send_file(method, str(chat_id), path, caption)
            return True
        except Exception as exc:
            self.connected = False
            self.on_status(f"Telegram upload failed ({type(exc).__name__})")
            return False


class TelegramTransport:
    """Telegram HTTP transport. Token is private and never included in errors."""
    API = "https://api.telegram.org"
    FILES = "https://api.telegram.org/file/bot"

    def __init__(self, token: str, timeout: float = 30.0) -> None:
        self.__token = token
        self.timeout = timeout

    def _request(self, url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> Any:
        request = Request(url, data=data, headers=headers or {})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            raise TelegramError("Telegram API request failed") from None
        if not payload.get("ok"):
            raise TelegramError("Telegram API rejected the request")
        return payload.get("result")

    def call(self, method: str, payload: dict[str, Any]) -> Any:
        data = urlencode(payload).encode("utf-8")
        return self._request(f"{self.API}/bot{self.__token}/{method}", data, {"Content-Type": "application/x-www-form-urlencoded"})

    def get_updates(self, offset: int, timeout: int = 20) -> list[dict[str, Any]]:
        result = self.call("getUpdates", {"offset": offset, "timeout": timeout, "allowed_updates": json.dumps(["message"])})
        return result if isinstance(result, list) else []

    def get_file(self, file_id: str) -> dict[str, Any]:
        return self.call("getFile", {"file_id": file_id})

    def download_file(self, file_path: str) -> bytes:
        url = f"{self.FILES}{self.__token}/{file_path}"
        try:
            with urlopen(url, timeout=self.timeout) as response:
                payload = response.read(20 * 1024 * 1024 + 1)
                if len(payload) > 20 * 1024 * 1024:
                    raise TelegramError("Photo exceeds the local processing size limit")
                return payload
        except (HTTPError, URLError, TimeoutError, OSError):
            raise TelegramError("Telegram file download failed") from None

    def send_file(self, method: str, chat_id: str, path: str, caption: str = "") -> Any:
        import os
        boundary = f"motionvision{time.time_ns():x}"
        field = "photo" if method == "sendPhoto" else "video"
        filename = os.path.basename(path)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode(),
        ]
        with open(path, "rb") as source:
            parts.extend((source.read(), b"\r\n"))
        parts.append(f"--{boundary}--\r\n".encode())
        return self._request(f"{self.API}/bot{self.__token}/{method}", b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"})
