import os
import sys
import unittest
from numpy import ndarray
from typing import Dict, List, Tuple

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from swarm_project_modules import TiledEnvironment, SwarmAgent, swarm_agent


class swarm_agent_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.swarm_agent = SwarmAgent(id=1)
        self.tiled_enviro = TiledEnvironment(
            height=5, width=5, ratio_of_white_to_black_tiles=0.5, clustered=False
        )

    def test_visited_cell_is_visited(self):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.assertEqual(
            self.swarm_agent.current_cell, (0, 0), "current cell not correct"
        )

    def test_visit_cell_again_cause_no_error(self):
        with self.subTest():
            self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
            self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])

    def test_occupied_cell_returns_is_occupied(self):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.assertEqual(
            self.tiled_enviro.tile_grid[(0, 0)]["occupied"],
            True,
            "occupation status not correct",
        )

    def test_another_member_visting_cell_cannot_enter_it(self):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        second_swarm_agent = SwarmAgent(id=2)
        second_swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.assertNotEqual(
            second_swarm_agent.current_cell, (0, 0), "occupied cell visit again"
        )

    def test_leaving_cell_makes_it_no_longer_occupied(self):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.swarm_agent.leave_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.assertEqual(
            self.tiled_enviro.tile_grid[(0, 0)]["occupied"],
            False,
            "occupation status not correct on leaving",
        )

    def test_occupy_leave_cell_allows_other_bot_to_occupy(self):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.swarm_agent.leave_cell(self.tiled_enviro.tile_grid[(0, 0)])
        second_swarm_agent = SwarmAgent(id=2)
        second_swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        with self.subTest():
            self.assertEqual(
                self.tiled_enviro.tile_grid[(0, 0)]["occupied"],
                True,
                "occupation status not correct on re-entry",
            )
            self.assertEqual(
                second_swarm_agent.current_cell,
                (0, 0),
                "cell not added to visited list",
            )

    def test_when_leaving_cell_it_is_added_to_visited_list(self):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.swarm_agent.leave_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.assertIn(
            (0, 0),
            self.swarm_agent.cells_visited,
            "cell when left not added to visited list",
        )

    def test_current_cell_not_in_visited_cell_return_reward_of_1(self):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.swarm_agent.leave_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(1, 0)])
        self.assertEqual(
            self.swarm_agent.return_reward(),
            1,
            "postive reward not returning correctly",
        )

    def test_current_cell_in_visited_cell_return_reward_of_0(self):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.swarm_agent.leave_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.assertEqual(
            self.swarm_agent.return_reward(),
            0,
            "zero reward not returning correctly",
        )


if __name__ == "__main__":
    unittest.main()
