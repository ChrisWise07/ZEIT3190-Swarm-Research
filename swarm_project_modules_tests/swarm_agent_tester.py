import os
import sys
import unittest
from numpy import ndarray
from typing import Dict, List, Tuple


ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from swarm_project_modules import TiledEnvironment, SwarmAgent, swarm_agent, Direction


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
            second_swarm_agent.current_cell,
            (0, 0),
            "visiting cell that is occupied not working",
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

    def succesful_forward_step_tester(
        self,
        new_cell: Tuple[int, int],
        agent_direction: Direction = Direction.RIGHT,
        starting_cell: Tuple[int, int] = (0, 0),
    ):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[starting_cell])
        self.swarm_agent.current_direction_facing = agent_direction
        self.swarm_agent.forward_step(self.tiled_enviro.tile_grid)
        with self.subTest():
            self.assertEqual(
                self.swarm_agent.current_cell,
                new_cell,
                f"current cell for {agent_direction} facing forward step not working",
            )
            self.assertIn(
                starting_cell,
                self.swarm_agent.cells_visited,
                f"cells visited for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.tiled_enviro.tile_grid[starting_cell]["occupied"],
                False,
                f"occupation for cell left for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.tiled_enviro.tile_grid[new_cell]["occupied"],
                True,
                f"occupation for current cell for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.swarm_agent.return_reward(),
                1,
                "correct reward not returning after succesful step forward",
            )

    def unsuccesful_forward_step_tester(
        self,
        current_cell: Tuple[int, int],
        pre_occupied_cell: Tuple[int, int],
        agent_direction: Direction = Direction.RIGHT,
        starting_cell: Tuple[int, int] = (0, 0),
    ):
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[starting_cell])
        self.swarm_agent.current_direction_facing = agent_direction
        self.swarm_agent.forward_step(self.tiled_enviro.tile_grid)
        with self.subTest():
            self.assertEqual(
                self.swarm_agent.current_cell,
                current_cell,
                f"current cell for {agent_direction} facing forward step not working",
            )
            self.assertIn(
                current_cell,
                self.swarm_agent.cells_visited,
                f"cells visited for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.tiled_enviro.tile_grid[current_cell]["occupied"],
                True,
                f"occupation for current cell for {agent_direction} facing forward step not working",
            )
            self.assertNotIn(
                pre_occupied_cell,
                self.swarm_agent.cells_visited,
                "previously occupied cell in visited cells list",
            )
            self.assertEqual(
                self.swarm_agent.return_reward(),
                0,
                "correct reward not returning after unsuccesful step forward",
            )

    def test_step_forward_agent_facing_right(self):
        self.succesful_forward_step_tester(new_cell=(0, 1))

    def test_step_forward_agent_facing_down(self):
        self.succesful_forward_step_tester(
            new_cell=(1, 0), agent_direction=Direction.DOWN
        )

    def test_step_forward_from_tile_0_1_agent_facing_left(self):
        self.succesful_forward_step_tester(
            new_cell=(0, 0), agent_direction=Direction.LEFT, starting_cell=(0, 1)
        )

    def test_step_forward_from_tile_1_0_agent_facing_up(self):
        self.succesful_forward_step_tester(
            new_cell=(0, 0), agent_direction=Direction.UP, starting_cell=(1, 0)
        )

    def test_step_forward_on_to_occupied_tile(self):
        second_swarm_agent = SwarmAgent(id=2)
        second_swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 1)])
        self.unsuccesful_forward_step_tester(
            current_cell=(0, 0), pre_occupied_cell=(0, 1)
        )


if __name__ == "__main__":
    unittest.main()
