"""Window geometry validation and small native window-manager bridge."""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkArea:
    x: int
    y: int
    width: int
    height: int
    primary: bool = False

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class WindowRect:
    x: int
    y: int
    width: int
    height: int


FALLBACK_WORK_AREA = WorkArea(0, 0, 1920, 1080, True)
DEFAULT_WINDOW_SIZE = (1280, 720)
MIN_WINDOW_SIZE = (320, 240)
MAX_WINDOW_SIZE = (7680, 4320)


def _primary(areas: tuple[WorkArea, ...]) -> WorkArea:
    return next((area for area in areas if area.primary), areas[0])


def _intersection_area(rect: WindowRect, area: WorkArea) -> int:
    width = max(0, min(rect.x + rect.width, area.right) - max(rect.x, area.x))
    height = max(0, min(rect.y + rect.height, area.bottom) - max(rect.y, area.y))
    return width * height


def normalize_window_rect(
    rect: WindowRect | tuple[int, int, int, int] | None,
    work_areas: tuple[WorkArea, ...] | list[WorkArea],
) -> WindowRect:
    """Return a sane, fully visible rectangle on the nearest useful monitor."""
    areas = tuple(area for area in work_areas if area.width > 0 and area.height > 0)
    if not areas:
        areas = (FALLBACK_WORK_AREA,)

    candidate: WindowRect | None
    if isinstance(rect, WindowRect):
        candidate = rect
    elif rect is None:
        candidate = None
    else:
        try:
            values = tuple(int(value) for value in rect)
            if len(values) != 4:
                raise ValueError
            candidate = WindowRect(values[0], values[1], values[2], values[3])
        except (TypeError, ValueError, OverflowError):
            candidate = None
    if candidate is None or candidate.width <= 0 or candidate.height <= 0:
        candidate = centered_window_rect(areas)

    overlaps = [(area, _intersection_area(candidate, area)) for area in areas]
    max_overlap = max(overlap for _, overlap in overlaps)
    if max_overlap:
        area = next(area for area, overlap in overlaps if overlap == max_overlap)
    else:
        area = _primary(areas)

    max_width = min(MAX_WINDOW_SIZE[0], area.width)
    max_height = min(MAX_WINDOW_SIZE[1], area.height)
    min_width = min(MIN_WINDOW_SIZE[0], max_width)
    min_height = min(MIN_WINDOW_SIZE[1], max_height)
    width = min(max(candidate.width, min_width), max_width)
    height = min(max(candidate.height, min_height), max_height)
    x = min(max(candidate.x, area.x), area.right - width)
    y = min(max(candidate.y, area.y), area.bottom - height)
    return WindowRect(x, y, width, height)


def centered_window_rect(
    work_areas: tuple[WorkArea, ...] | list[WorkArea],
    size: tuple[int, int] = DEFAULT_WINDOW_SIZE,
) -> WindowRect:
    areas = tuple(area for area in work_areas if area.width > 0 and area.height > 0)
    if not areas:
        areas = (FALLBACK_WORK_AREA,)
    area = _primary(areas)
    width = min(max(int(size[0]), min(MIN_WINDOW_SIZE[0], area.width)), min(MAX_WINDOW_SIZE[0], area.width))
    height = min(max(int(size[1]), min(MIN_WINDOW_SIZE[1], area.height)), min(MAX_WINDOW_SIZE[1], area.height))
    return WindowRect(area.x + (area.width - width) // 2, area.y + (area.height - height) // 2, width, height)


class WindowManager:
    """Read Windows monitor/window state; degrade safely on other systems."""

    SW_RESTORE = 9
    SW_MAXIMIZE = 3
    VK_CONTROL = 0x11
    VK_SHIFT = 0x10

    def __init__(self) -> None:
        self._user32: Any = None
        if os.name == "nt":
            try:
                self._user32 = ctypes.WinDLL("user32", use_last_error=True)
            except (AttributeError, OSError):
                self._user32 = None

    def work_areas(self) -> tuple[WorkArea, ...]:
        if self._user32 is None:
            return (FALLBACK_WORK_AREA,)
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = (("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long))

        class MONITORINFO(ctypes.Structure):
            _fields_ = (("cbSize", ctypes.c_ulong), ("rcMonitor", RECT),
                        ("rcWork", RECT), ("dwFlags", ctypes.c_ulong))

        monitor_proc = ctypes.WINFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_ssize_t
        )
        try:
            self._user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(MONITORINFO))
            self._user32.GetMonitorInfoW.restype = wintypes.BOOL
            self._user32.EnumDisplayMonitors.argtypes = (
                wintypes.HDC, ctypes.POINTER(RECT), monitor_proc, wintypes.LPARAM
            )
            self._user32.EnumDisplayMonitors.restype = wintypes.BOOL
        except (AttributeError, OSError, TypeError):
            pass
        results: list[WorkArea] = []

        def collect(monitor: int, _dc: int, _rect: Any, _data: int) -> int:
            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(info)
            if self._user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                work = info.rcWork
                if work.right > work.left and work.bottom > work.top:
                    results.append(WorkArea(work.left, work.top, work.right - work.left,
                                            work.bottom - work.top, bool(info.dwFlags & 1)))
            return 1

        callback = monitor_proc(collect)
        try:
            self._user32.EnumDisplayMonitors(None, None, callback, 0)
        except (AttributeError, OSError, ctypes.ArgumentError):
            return (FALLBACK_WORK_AREA,)
        return tuple(results) or (FALLBACK_WORK_AREA,)

    def client_work_areas(self, window_name: str) -> tuple[WorkArea, ...]:
        """Return work areas inset by this window's non-client frame."""
        areas = self.work_areas()
        handle = self._window_handle(window_name)
        if handle is None:
            return areas
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = (("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long))

        try:
            self._user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(RECT))
            self._user32.GetWindowRect.restype = wintypes.BOOL
            self._user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(RECT))
            self._user32.GetClientRect.restype = wintypes.BOOL
            self._user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
            self._user32.ClientToScreen.restype = wintypes.BOOL
            outer, client = RECT(), RECT()
            if not self._user32.GetWindowRect(handle, ctypes.byref(outer)):
                return areas
            if not self._user32.GetClientRect(handle, ctypes.byref(client)):
                return areas
            client_origin = wintypes.POINT(client.left, client.top)
            client_end = wintypes.POINT(client.right, client.bottom)
            if not self._user32.ClientToScreen(handle, ctypes.byref(client_origin)):
                return areas
            if not self._user32.ClientToScreen(handle, ctypes.byref(client_end)):
                return areas
        except (AttributeError, OSError, TypeError, ctypes.ArgumentError):
            return areas

        left = client_origin.x - outer.left
        top = client_origin.y - outer.top
        right = outer.right - client_end.x
        bottom = outer.bottom - client_end.y
        if min(left, top, right, bottom) < 0:
            return areas
        return tuple(
            WorkArea(area.x + left, area.y + top,
                     max(1, area.width - left - right),
                     max(1, area.height - top - bottom), area.primary)
            for area in areas
        )

    def move_to_client_origin(self, window_name: str, x: int, y: int) -> bool:
        """Place the client image origin at screen coordinates x/y."""
        handle = self._window_handle(window_name)
        if handle is None:
            return False
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = (("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long))

        try:
            self._user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(RECT))
            self._user32.GetWindowRect.restype = wintypes.BOOL
            self._user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(RECT))
            self._user32.GetClientRect.restype = wintypes.BOOL
            self._user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
            self._user32.ClientToScreen.restype = wintypes.BOOL
            self._user32.SetWindowPos.argtypes = (
                wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, wintypes.UINT,
            )
            self._user32.SetWindowPos.restype = wintypes.BOOL
            outer, client = RECT(), RECT()
            if not self._user32.GetWindowRect(handle, ctypes.byref(outer)):
                return False
            if not self._user32.GetClientRect(handle, ctypes.byref(client)):
                return False
            origin = wintypes.POINT(client.left, client.top)
            if not self._user32.ClientToScreen(handle, ctypes.byref(origin)):
                return False
            offset_x, offset_y = origin.x - outer.left, origin.y - outer.top
            flags = 0x0001 | 0x0004 | 0x0010  # SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
            return bool(self._user32.SetWindowPos(handle, None, x - offset_x, y - offset_y, 0, 0, flags))
        except (AttributeError, OSError, TypeError, ctypes.ArgumentError):
            return False

    def _window_handle(self, window_name: str) -> int | None:
        if self._user32 is None:
            return None
        from ctypes import wintypes
        try:
            self._user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
            self._user32.FindWindowW.restype = wintypes.HWND
            handle = self._user32.FindWindowW(None, window_name)
            return int(handle) if handle else None
        except (AttributeError, OSError, TypeError):
            return None

    def state(self, window_name: str) -> str | None:
        handle = self._window_handle(window_name)
        if handle is None:
            return None
        from ctypes import wintypes
        try:
            self._user32.IsIconic.argtypes = (wintypes.HWND,)
            self._user32.IsIconic.restype = wintypes.BOOL
            self._user32.IsZoomed.argtypes = (wintypes.HWND,)
            self._user32.IsZoomed.restype = wintypes.BOOL
            if self._user32.IsIconic(handle):
                return "minimized"
            if self._user32.IsZoomed(handle):
                return "maximized"
            return "normal"
        except (AttributeError, OSError, TypeError):
            return None

    def restore(self, window_name: str) -> None:
        handle = self._window_handle(window_name)
        if handle is not None:
            try:
                from ctypes import wintypes
                self._user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
                self._user32.ShowWindow.restype = wintypes.BOOL
                self._user32.ShowWindow(handle, self.SW_RESTORE)
            except (AttributeError, OSError, TypeError):
                pass

    def maximize(self, window_name: str) -> None:
        handle = self._window_handle(window_name)
        if handle is not None:
            try:
                from ctypes import wintypes
                self._user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
                self._user32.ShowWindow.restype = wintypes.BOOL
                self._user32.ShowWindow(handle, self.SW_MAXIMIZE)
            except (AttributeError, OSError, TypeError):
                pass

    def reset_shortcut(self, key: int) -> bool:
        if key & 0xFF not in (ord("r"), ord("R")) or self._user32 is None:
            return False
        try:
            self._user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
            self._user32.GetAsyncKeyState.restype = ctypes.c_short
            control = bool(self._user32.GetAsyncKeyState(self.VK_CONTROL) & 0x8000)
            shift = bool(self._user32.GetAsyncKeyState(self.VK_SHIFT) & 0x8000)
            return control and shift
        except (AttributeError, OSError, TypeError):
            return False
