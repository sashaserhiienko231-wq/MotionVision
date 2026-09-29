from __future__ import annotations

import unittest

from windowing import (
    WindowRect,
    WorkArea,
    centered_window_rect,
    normalize_window_rect,
)


class WindowGeometryTests(unittest.TestCase):
    def test_offscreen_geometry_is_moved_to_primary_work_area(self):
        areas = (
            WorkArea(0, 0, 1920, 1040, True),
            WorkArea(1920, -200, 1280, 1000),
        )
        safe = normalize_window_rect(WindowRect(-5000, 3000, 1200, 800), areas)
        self.assertEqual(safe, WindowRect(0, 240, 1200, 800))

    def test_invalid_and_oversized_dimensions_are_repaired(self):
        area = WorkArea(-1280, 0, 1280, 720, True)
        self.assertEqual(
            normalize_window_rect(WindowRect(9000, 9000, -1, 0), (area,)),
            WindowRect(-1280, 0, 1280, 720),
        )
        self.assertEqual(
            normalize_window_rect(WindowRect(-100, 20, 10000, 6000), (area,)),
            WindowRect(-1280, 0, 1280, 720),
        )

    def test_monitor_work_area_change_clamps_geometry_again(self):
        expanded = (WorkArea(0, 0, 1920, 1080, True),)
        reduced = (WorkArea(0, 0, 1280, 720, True),)
        saved = WindowRect(1500, 700, 1000, 600)
        self.assertEqual(normalize_window_rect(saved, expanded), WindowRect(920, 480, 1000, 600))
        self.assertEqual(normalize_window_rect(saved, reduced), WindowRect(280, 120, 1000, 600))

    def test_reset_geometry_is_centered_and_fits_selected_primary(self):
        areas = (
            WorkArea(0, 0, 1280, 720),
            WorkArea(-1600, 100, 1600, 900, True),
        )
        self.assertEqual(centered_window_rect(areas), WindowRect(-1440, 190, 1280, 720))


if __name__ == "__main__":
    unittest.main()
