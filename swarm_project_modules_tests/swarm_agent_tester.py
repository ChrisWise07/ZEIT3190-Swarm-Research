import os
import sys
import unittest
from typing import Dict, List, Tuple, Union

from black import diff


ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from swarm_project_modules import (
    TiledEnvironment,
    SwarmAgent,
    Direction,
    Turn,
    ObjectType,
    RelativePosition,
    RelativeMotion,
)


class swarm_agent_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tiled_enviro = TiledEnvironment(
            height=5, width=5, ratio_of_white_to_black_tiles=0.5, clustered=False
        )
        self.swarm_agent = SwarmAgent(
            id=1, starting_cell=self.tiled_enviro.tile_grid[(0, 0)]
        )

    def test_visited_cell_is_visited(self):
        self.assertEqual(
            self.swarm_agent.current_cell, (0, 0), "current cell not correct"
        )

    def test_visit_cell_again_cause_no_error(self):
        with self.subTest():
            self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(0, 0)])

    def test_occupied_cell_returns_is_occupied(self):
        self.assertEqual(
            self.tiled_enviro.tile_grid[(0, 0)]["occupied"],
            True,
            "occupation status not correct",
        )

    def test_another_member_visting_cell_cannot_enter_it(self):
        second_swarm_agent = SwarmAgent(
            id=2, starting_cell=self.tiled_enviro.tile_grid[(0, 0)]
        )
        self.assertNotEqual(
            second_swarm_agent.current_cell,
            (0, 0),
            "visiting cell that is occupied not working",
        )

    def test_leaving_cell_makes_it_no_longer_occupied(self):
        self.swarm_agent.leave_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.assertEqual(
            self.tiled_enviro.tile_grid[(0, 0)]["occupied"],
            False,
            "occupation status not correct on leaving",
        )

    def test_occupy_leave_cell_allows_other_agent_to_occupy(self):
        self.swarm_agent.leave_cell(self.tiled_enviro.tile_grid[(0, 0)])
        second_swarm_agent = SwarmAgent(
            id=2, starting_cell=self.tiled_enviro.tile_grid[(0, 0)]
        )
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
        self.swarm_agent.leave_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.assertIn(
            (0, 0),
            self.swarm_agent.cells_visited,
            "cell when left not added to visited list",
        )

    def test_current_cell_not_in_visited_cell_return_reward_of_1(self):
        self.swarm_agent.leave_cell(self.tiled_enviro.tile_grid[(0, 0)])
        self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[(1, 0)])
        self.assertEqual(
            self.swarm_agent.return_reward(),
            1,
            "postive reward not returning correctly",
        )

    def test_current_cell_in_visited_cell_return_reward_of_0(self):
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
        different_starting_cell: Dict = None,
    ):
        if different_starting_cell:
            self.swarm_agent.leave_cell(
                self.tiled_enviro.tile_grid[self.swarm_agent.current_cell]
            )
            self.swarm_agent = SwarmAgent(
                id=1, starting_cell=self.tiled_enviro.tile_grid[different_starting_cell]
            )

        starting_cell = self.swarm_agent.current_cell

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

    def test_step_forward_agent_facing_right(self):
        self.succesful_forward_step_tester(new_cell=(0, 1))

    def test_step_forward_agent_facing_down(self):
        self.succesful_forward_step_tester(
            new_cell=(1, 0), agent_direction=Direction.DOWN
        )

    def test_step_forward_from_tile_0_1_agent_facing_left(self):
        self.succesful_forward_step_tester(
            new_cell=(0, 0),
            agent_direction=Direction.LEFT,
            different_starting_cell=(0, 1),
        )

    def test_step_forward_from_tile_1_0_agent_facing_up(self):
        self.succesful_forward_step_tester(
            new_cell=(0, 0),
            agent_direction=Direction.UP,
            different_starting_cell=(1, 0),
        )

    def unsuccesful_forward_step_tester(
        self,
        starting_cell: Tuple[int, int] = (0, 0),
        agent_direction: Direction = Direction.RIGHT,
        pre_occupied_cell: Tuple[int, int] = None,
    ):
        if not (
            self.swarm_agent.current_cell
            == self.tiled_enviro.tile_grid[starting_cell]["id"]
        ):
            self.swarm_agent.leave_cell(
                self.tiled_enviro.tile_grid[self.swarm_agent.current_cell]
            )
            self.swarm_agent = SwarmAgent(
                id=1, starting_cell=self.tiled_enviro.tile_grid[starting_cell]
            )

        self.swarm_agent.current_direction_facing = agent_direction

        self.swarm_agent.forward_step(self.tiled_enviro.tile_grid)

        with self.subTest():
            self.assertEqual(
                self.swarm_agent.current_cell,
                starting_cell,
                f"current cell for {agent_direction} facing forward step not working",
            )
            self.assertIn(
                starting_cell,
                self.swarm_agent.cells_visited,
                f"cells visited for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.tiled_enviro.tile_grid[starting_cell]["occupied"],
                True,
                f"occupation for current cell for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.swarm_agent.return_reward(),
                0,
                "correct reward not returning after unsuccesful step forward",
            )
            if pre_occupied_cell:
                self.assertNotIn(
                    pre_occupied_cell,
                    self.swarm_agent.cells_visited,
                    "previously occupied cell in visited cells list",
                )

    def test_step_forward_on_to_occupied_tile(self):
        SwarmAgent(id=2, starting_cell=self.tiled_enviro.tile_grid[(0, 1)])
        self.unsuccesful_forward_step_tester(pre_occupied_cell=(0, 1))

    def wall_and_corner_tester(
        self, directions: List[Direction], starting_cell: Tuple[int, int] = (0, 0)
    ) -> None:
        for direction in directions:
            self.unsuccesful_forward_step_tester(
                starting_cell=starting_cell,
                agent_direction=direction,
            )

    def test_step_forward_into_top_left_corner(self):
        self.wall_and_corner_tester(directions=[Direction.LEFT, Direction.UP])

    def test_step_forward_into_top_right_corner(self):
        self.wall_and_corner_tester(
            starting_cell=(0, 4), directions=[Direction.RIGHT, Direction.UP]
        )

    def test_step_forward_into_bottom_left_corner(self):
        self.wall_and_corner_tester(
            starting_cell=(4, 0), directions=[Direction.LEFT, Direction.DOWN]
        )

    def test_step_forward_into_bottom_right_corner(self):
        self.wall_and_corner_tester(
            starting_cell=(4, 4), directions=[Direction.RIGHT, Direction.DOWN]
        )

    def test_step_forward_into_top_wall(self):
        self.wall_and_corner_tester(starting_cell=(0, 1), directions=[Direction.UP])

    def test_step_forward_into_left_wall(self):
        self.wall_and_corner_tester(starting_cell=(1, 0), directions=[Direction.LEFT])

    def test_step_forward_into_right_wall(self):
        self.wall_and_corner_tester(starting_cell=(1, 4), directions=[Direction.RIGHT])

    def test_step_forward_into_bottom_wall(self):
        self.wall_and_corner_tester(starting_cell=(4, 1), directions=[Direction.DOWN])

    def rotation_tester(
        self,
        rotations: List[Turn],
        correction_directions: List[Direction],
        starting_direction: Direction = Direction.UP,
    ) -> None:
        self.swarm_agent.current_direction_facing = starting_direction
        for rotation, correction_direction in zip(rotations, correction_directions):
            self.swarm_agent.turn(turn_type=rotation)
            self.assertEqual(
                self.swarm_agent.current_direction_facing,
                correction_direction,
                f"Turning to the {rotation} to ended up in {correction_direction} didn't work",
            )

    def test_from_up_positon_turn_right_turn_left(self):
        self.rotation_tester(
            rotations=[Turn.RIGHT, Turn.LEFT],
            correction_directions=[Direction.RIGHT, Direction.UP],
        )

    def test_from_down_positon_turn_left_turn_right(self):
        self.rotation_tester(
            starting_direction=Direction.DOWN,
            rotations=[Turn.LEFT, Turn.RIGHT],
            correction_directions=[Direction.RIGHT, Direction.DOWN],
        )

    def test_from_up_positon_full_clockwise_turn(self):
        self.rotation_tester(
            rotations=[Turn.RIGHT] * 4,
            correction_directions=[
                Direction.RIGHT,
                Direction.DOWN,
                Direction.LEFT,
                Direction.UP,
            ],
        )

    def test_from_up_positon_full_anticlockwise_turn(self):
        self.rotation_tester(
            rotations=[Turn.LEFT] * 4,
            correction_directions=[
                Direction.LEFT,
                Direction.DOWN,
                Direction.RIGHT,
                Direction.UP,
            ],
        )

    def object_type_tester(
        self,
        correct_object: ObjectType,
        error_message: str,
        direction: Direction = Direction.UP,
        cell: Tuple[int, int] = None,
    ) -> None:
        if cell:
            self.swarm_agent.leave_cell(
                self.tiled_enviro.tile_grid[self.swarm_agent.current_cell]
            )
            self.swarm_agent.occupy_cell(self.tiled_enviro.tile_grid[cell])

        self.swarm_agent.current_direction_facing = direction

        nearest_object = self.swarm_agent.get_type_of_nearest_object(
            tile_grid=self.tiled_enviro.tile_grid
        )

        self.assertEqual(nearest_object, correct_object, error_message)

    def test_top_left_corner_is_detected_when_on_that_tile(self):
        self.object_type_tester(
            correct_object=ObjectType.CORNER,
            error_message="Top left corner not detected",
        )

    def test_top_right_corner_is_detected_when_on_that_tile(self):
        self.object_type_tester(
            cell=(0, 4),
            correct_object=ObjectType.CORNER,
            error_message="Top right corner not detected",
        )

    def test_top_wall_is_detected_when_on_that_tile(self):
        self.object_type_tester(
            cell=(0, 1),
            correct_object=ObjectType.WALL,
            error_message="Top wall not detected",
        )

    def test_bottom_wall_is_detected_when_on_that_tile(self):
        self.object_type_tester(
            cell=(4, 1),
            correct_object=ObjectType.WALL,
            error_message="Top wall not detected",
        )

    def test_no_wall_is_detected_when_on_that_tile(self):
        self.object_type_tester(
            cell=(2, 2),
            correct_object=ObjectType.NONE,
            error_message="no walls not detected",
        )

    def test_top_wall_is_detected_when_one_tile_below(self):
        self.object_type_tester(
            cell=(1, 1),
            correct_object=ObjectType.WALL,
            direction=Direction.UP,
            error_message="Top wall not detected when one tile below",
        )

    def test_right_wall_is_detected_when_one_tile_left(self):
        self.object_type_tester(
            cell=(1, 3),
            correct_object=ObjectType.WALL,
            direction=Direction.RIGHT,
            error_message="right wall not detected when one tile right",
        )

    def test_agent_detect_when_facing_occupied_tile(self):
        SwarmAgent(id=2, starting_cell=self.tiled_enviro.tile_grid[(0, 1)])

        self.object_type_tester(
            cell=(1, 1),
            correct_object=ObjectType.AGENT,
            direction=Direction.UP,
            error_message="Not detecting agent when one below one on empty tile",
        )

    def test_agent_doesnt_detect_when_not_facing_occupied_tile(self):
        SwarmAgent(id=2, starting_cell=self.tiled_enviro.tile_grid[(0, 1)])

        self.object_type_tester(
            cell=(1, 1),
            correct_object=ObjectType.NONE,
            direction=Direction.DOWN,
            error_message=(
                "Detecting agent when one below on empty tile but not facing agent"
            ),
        )

    def relative_postion_motion_tester(
        self,
        func,
        correct_relative_position_or_motion: Union[RelativePosition, RelativeMotion],
        obj_description: str,
        different_starting_cell: Tuple[int, int] = None,
        agent_direction: Direction = Direction.RIGHT,
    ) -> None:
        if different_starting_cell:
            self.swarm_agent.leave_cell(
                self.tiled_enviro.tile_grid[self.swarm_agent.current_cell]
            )
            self.swarm_agent.occupy_cell(
                self.tiled_enviro.tile_grid[different_starting_cell]
            )

        self.swarm_agent.current_direction_facing = agent_direction

        current_relative_position_or_motion = func(
            tile=self.tiled_enviro.tile_grid[self.swarm_agent.current_cell]
        )

        self.assertEqual(
            current_relative_position_or_motion,
            correct_relative_position_or_motion,
            (
                f"Didn't return {obj_description} is {correct_relative_position_or_motion} when facing {agent_direction}"
            ),
        )

    def test_agent_returns_left_of_top_left_corner_when_facing_right(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativePosition.LEFT,
            func=self.swarm_agent.get_relative_position_of_object,
            obj_description="top left corner",
        )

    def test_agent_returns_right_of_top_left_corner_when_facing_left(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativePosition.RIGHT,
            func=self.swarm_agent.get_relative_position_of_object,
            obj_description="top left corner",
            agent_direction=Direction.LEFT,
        )

    def test_agent_returns_left_of_top_left_corner_when_facing_up(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativePosition.LEFT,
            func=self.swarm_agent.get_relative_position_of_object,
            obj_description="top left corner",
            agent_direction=Direction.UP,
        )

    def test_agent_returns_right_of_top_left_corner_when_facing_down(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativePosition.RIGHT,
            func=self.swarm_agent.get_relative_position_of_object,
            obj_description="top left corner",
            agent_direction=Direction.DOWN,
        )

    def test_agent_returns_top_wall_is_behind_when_facing_down(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativePosition.BEHIND,
            func=self.swarm_agent.get_relative_position_of_object,
            obj_description="top wall",
            different_starting_cell=(0, 1),
            agent_direction=Direction.DOWN,
        )

    def test_agent_returns_bottom_left_corner_is_right_when_facing_up(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativePosition.RIGHT,
            func=self.swarm_agent.get_relative_position_of_object,
            obj_description="bottom left corner",
            different_starting_cell=(4, 4),
            agent_direction=Direction.UP,
        )

    def test_agent_returns_bottom_wall_is_front_when_facing_down(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativePosition.FRONT,
            func=self.swarm_agent.get_relative_position_of_object,
            obj_description="bottom wall",
            different_starting_cell=(4, 1),
            agent_direction=Direction.DOWN,
        )

    def test_agent_is_approaching_corner_when_facing_in_to_it(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativeMotion.APPROACHING,
            func=self.swarm_agent.get_relative_motion_of_object,
            obj_description="top left corner",
            agent_direction=Direction.UP,
        )

    def test_agent_is_escaping_corner_when_facing_away_from_it(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativeMotion.ESCAPING,
            func=self.swarm_agent.get_relative_motion_of_object,
            obj_description="top left corner",
            agent_direction=Direction.DOWN,
        )

    def test_agent_is_escaping_from_top_wall_when_right_to_wall(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativeMotion.ESCAPING,
            func=self.swarm_agent.get_relative_motion_of_object,
            obj_description="top wall",
            different_starting_cell=(0, 1),
            agent_direction=Direction.RIGHT,
        )

    def test_agent_is_escaping_from_top_wall_when_left_to_wall(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativeMotion.ESCAPING,
            func=self.swarm_agent.get_relative_motion_of_object,
            obj_description="top wall",
            different_starting_cell=(0, 1),
            agent_direction=Direction.LEFT,
        )

    def test_agent_is_escaping_from_bottom_wall_when_infront_of_wall(self):
        self.relative_postion_motion_tester(
            correct_relative_position_or_motion=RelativeMotion.ESCAPING,
            func=self.swarm_agent.get_relative_motion_of_object,
            obj_description="bottom wall",
            different_starting_cell=(4, 1),
            agent_direction=Direction.UP,
        )


if __name__ == "__main__":
    unittest.main()
