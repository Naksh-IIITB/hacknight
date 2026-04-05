import unittest

from f1_ai import F1Predictor


class PredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = F1Predictor()

    def test_qualifying_prediction_shape(self) -> None:
        result = self.predictor.predict_qualifying("mercedes", "australia", driver_number=12)
        self.assertIn("predicted_lap_time", result)
        self.assertIn("model_blend", result)
        self.assertEqual(result["driver"], "Kimi Antonelli")
        self.assertGreater(result["predicted_lap_time_sec"], 70.0)

    def test_race_simulation_stints_sum_to_race_distance(self) -> None:
        result = self.predictor.simulate_race("ferrari", "japan", driver_number=16)
        total_laps = sum(stint["laps"] for stint in result["stints"])
        self.assertEqual(total_laps, 53)
        self.assertIn("expected_race_result", result)

    def test_driver_specific_teammates_produce_different_outputs(self) -> None:
        verstappen = self.predictor.predict_qualifying("red_bull", "japan", driver_number=3)
        hadjar = self.predictor.predict_qualifying("red_bull", "japan", driver_number=6)
        self.assertNotEqual(verstappen["predicted_lap_time_sec"], hadjar["predicted_lap_time_sec"])

    def test_expected_race_order_contains_positions(self) -> None:
        order = self.predictor.expected_race_order("japan")
        self.assertGreaterEqual(len(order), 22)
        self.assertEqual(order[0]["position"], 1)

    def test_strategy_board_contains_all_teams(self) -> None:
        board = self.predictor.build_team_strategy_board("china")
        self.assertEqual(len(board), 11)

    def test_track_map_fallback_shape(self) -> None:
        track_map = self.predictor.get_track_map("japan")
        self.assertIn("polyline", track_map)
        self.assertIn("corners", track_map)

    def test_driver_grid_contains_22_entries_and_requested_drivers(self) -> None:
        drivers = self.predictor.list_drivers()
        self.assertEqual(len(drivers), 22)
        numbers = {driver["number"] for driver in drivers}
        self.assertIn(11, numbers)
        self.assertIn(77, numbers)


if __name__ == "__main__":
    unittest.main()
