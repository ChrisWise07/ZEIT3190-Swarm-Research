import unittest
from typing import List, Tuple, Union

from environment_agent_modules import (
    SwarmAgent,
    Direction,
    Turn,
    ObjectType,
    RelativePosition,
    create_nonclustered_tile_grid,
)


class swarm_agent_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tiled_enviro = create_nonclustered_tile_grid(
            height=5, width=5, ratio_of_white_to_black_tiles=0.5
        )
        self.swarm_agent = SwarmAgent(id=1, starting_cell=self.tiled_enviro[(0, 0)])

    def test_visited_cell_is_visited(self):
        self.assertEqual(
            self.swarm_agent.current_cell, (0, 0), "current cell not correct"
        )

    def test_visit_cell_again_cause_no_error(self):
        with self.subTest():
            self.swarm_agent.occupy_cell(self.tiled_enviro[(0, 0)])

    def test_occupied_cell_returns_is_occupied(self):
        self.assertEqual(
            self.tiled_enviro[(0, 0)]["occupied"],
            True,
            "occupation status not correct",
        )

    def test_another_member_visting_cell_cannot_enter_it(self):
        second_swarm_agent = SwarmAgent(id=2, starting_cell=self.tiled_enviro[(0, 0)])
        self.assertNotEqual(
            second_swarm_agent.current_cell,
            (0, 0),
            "visiting cell that is occupied not working",
        )

    def test_leaving_cell_makes_it_no_longer_occupied(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        self.assertEqual(
            self.tiled_enviro[(0, 0)]["occupied"],
            False,
            "occupation status not correct on leaving",
        )

    def test_occupy_leave_cell_allows_other_agent_to_occupy(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        second_swarm_agent = SwarmAgent(id=2, starting_cell=self.tiled_enviro[(0, 0)])
        with self.subTest():
            self.assertEqual(
                self.tiled_enviro[(0, 0)]["occupied"],
                True,
                "occupation status not correct on re-entry",
            )
            self.assertEqual(
                second_swarm_agent.current_cell,
                (0, 0),
                "cell not added to visited list",
            )

    def test_when_leaving_cell_it_is_added_to_visited_list(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        self.assertIn(
            (0, 0),
            self.swarm_agent.cells_visited,
            "cell when left not added to visited list",
        )

    def test_current_cell_not_in_visited_cell_return_reward_of_1(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        self.swarm_agent.occupy_cell(self.tiled_enviro[(1, 0)])
        self.assertEqual(
            self.swarm_agent.return_navigation_reward(),
            1,
            "postive reward not returning correctly",
        )

    def test_current_cell_in_visited_cell_return_reward_of_neg_1(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        self.swarm_agent.occupy_cell(self.tiled_enviro[(0, 0)])
        self.assertEqual(
            self.swarm_agent.return_navigation_reward(),
            -1,
            "zero reward not returning correctly",
        )

    def start_on_different_cell(
        self, different_starting_cell: Tuple[int, int]
    ) -> SwarmAgent:
        self.swarm_agent.leave_cell(self.tiled_enviro[self.swarm_agent.current_cell])
        return SwarmAgent(
            id=1, starting_cell=self.tiled_enviro[different_starting_cell]
        )

    def succesful_forward_step_tester(
        self,
        new_cell: Tuple[int, int],
        agent_direction: Direction = Direction.RIGHT,
        different_starting_cell: Tuple[int, int] = None,
    ):
        if different_starting_cell:
            self.swarm_agent = self.start_on_different_cell(
                different_starting_cell=different_starting_cell
            )

        starting_cell = self.swarm_agent.current_cell

        self.swarm_agent.current_direction_facing = agent_direction.value

        self.swarm_agent.forward_step(self.tiled_enviro)

        self.swarm_agent.add_cell_to_visited_list(starting_cell)

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
                self.tiled_enviro[starting_cell]["occupied"],
                False,
                f"occupation for cell left for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.tiled_enviro[new_cell]["occupied"],
                True,
                f"occupation for current cell for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.swarm_agent.return_navigation_reward(),
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
        different_starting_cell: Tuple[int, int] = None,
        agent_direction: Direction = Direction.RIGHT,
        pre_occupied_cell: Tuple[int, int] = None,
    ):
        if different_starting_cell:
            self.swarm_agent = self.start_on_different_cell(
                different_starting_cell=different_starting_cell
            )

        starting_cell = self.swarm_agent.current_cell

        self.swarm_agent.current_direction_facing = agent_direction.value

        self.swarm_agent.forward_step(self.tiled_enviro)

        self.swarm_agent.add_cell_to_visited_list(starting_cell)

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
                self.tiled_enviro[starting_cell]["occupied"],
                True,
                f"occupation for current cell for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.swarm_agent.return_navigation_reward(),
                -1,
                "correct reward not returning after unsuccesful step forward",
            )
            if pre_occupied_cell:
                self.assertNotIn(
                    pre_occupied_cell,
                    self.swarm_agent.cells_visited,
                    "previously occupied cell in visited cells list",
                )

    def test_step_forward_on_to_occupied_tile(self):
        SwarmAgent(id=2, starting_cell=self.tiled_enviro[(0, 1)])
        self.unsuccesful_forward_step_tester(pre_occupied_cell=(0, 1))

    def wall_and_corner_tester(
        self,
        directions: List[Direction],
        different_starting_cell: Tuple[int, int] = (0, 0),
    ) -> None:
        for direction in directions:
            self.unsuccesful_forward_step_tester(
                different_starting_cell=different_starting_cell,
                agent_direction=direction,
            )

    def test_step_forward_into_top_left_corner(self):
        self.wall_and_corner_tester(directions=[Direction.LEFT, Direction.UP])

    def test_step_forward_into_top_right_corner(self):
        self.wall_and_corner_tester(
            different_starting_cell=(0, 4), directions=[Direction.RIGHT, Direction.UP]
        )

    def test_step_forward_into_bottom_left_corner(self):
        self.wall_and_corner_tester(
            different_starting_cell=(4, 0), directions=[Direction.LEFT, Direction.DOWN]
        )

    def test_step_forward_into_bottom_right_corner(self):
        self.wall_and_corner_tester(
            different_starting_cell=(4, 4), directions=[Direction.RIGHT, Direction.DOWN]
        )

    def test_step_forward_into_top_wall(self):
        self.wall_and_corner_tester(
            different_starting_cell=(0, 1), directions=[Direction.UP]
        )

    def test_step_forward_into_left_wall(self):
        self.wall_and_corner_tester(
            different_starting_cell=(1, 0), directions=[Direction.LEFT]
        )

    def test_step_forward_into_right_wall(self):
        self.wall_and_corner_tester(
            different_starting_cell=(1, 4), directions=[Direction.RIGHT]
        )

    def test_step_forward_into_bottom_wall(self):
        self.wall_and_corner_tester(
            different_starting_cell=(4, 1), directions=[Direction.DOWN]
        )

    def rotation_tester(
        self,
        rotations: List[Turn],
        correction_directions: List[Direction],
        starting_direction: Direction = Direction.UP,
    ) -> None:
        self.swarm_agent.current_direction_facing = starting_direction.value
        for rotation, correction_direction in zip(rotations, correction_directions):
            self.swarm_agent.turn(turn_type=rotation.value)
            self.assertEqual(
                self.swarm_agent.current_direction_facing,
                correction_direction.value,
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

    def state_tester(
        self,
        correct_navigation_states: List[Union[ObjectType, RelativePosition]],
        obj_description: str,
        agent_direction: Direction = Direction.RIGHT,
        different_starting_cell: Tuple[int, int] = None,
    ):
        if different_starting_cell:
            self.swarm_agent = self.start_on_different_cell(
                different_starting_cell=different_starting_cell
            )

        self.swarm_agent.current_direction_facing = agent_direction.value

        current_navigation_states = self.swarm_agent.get_navigation_states(
            self.tiled_enviro
        )

        for current_state, correct_state in zip(
            current_navigation_states, correct_navigation_states
        ):
            self.assertEqual(
                current_state,
                correct_state.value,
                (
                    f"{correct_state} not returned when facing {agent_direction} at {self.swarm_agent.current_cell} observing {obj_description}"
                ),
            )

    def test_agent_returns_correct_state_facing_up_at_top_left_corner(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.CORNER,
                RelativePosition.LEFT_FRONT,
            ],
            obj_description="top left corner",
            agent_direction=Direction.UP,
        )

    def test_agent_returns_correct_state_facing_right_at_top_left_corner(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.WALL,
                RelativePosition.LEFT,
            ],
            obj_description="top left corner",
        )

    def test_agent_returns_correct_state_facing_down_at_top_left_corner(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.WALL,
                RelativePosition.RIGHT,
            ],
            obj_description="top left corner",
            agent_direction=Direction.DOWN,
        )

    def test_agent_returns_correct_state_facing_left_at_top_left_corner(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.CORNER,
                RelativePosition.RIGHT_FRONT,
            ],
            obj_description="top left corner",
            agent_direction=Direction.LEFT,
        )

    def test_agent_returns_correct_state_facing_down_at_top_wall(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.NONE,
                RelativePosition.FRONT,
            ],
            obj_description="top wall",
            agent_direction=Direction.DOWN,
            different_starting_cell=(0, 1),
        )

    def test_agent_returns_correct_state_facing_down_at_bottom_wall(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.WALL,
                RelativePosition.FRONT,
            ],
            obj_description="bottom wall",
            agent_direction=Direction.DOWN,
            different_starting_cell=(4, 1),
        )

    def test_agent_returns_correct_state_facing_away_from_left_wall(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.NONE,
                RelativePosition.FRONT,
            ],
            obj_description="left wall",
            agent_direction=Direction.RIGHT,
            different_starting_cell=(1, 0),
        )

    def test_agent_returns_correct_state_facing_up_left_of_right_wall(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.CORNER,
                RelativePosition.RIGHT_FRONT,
            ],
            obj_description="right wall",
            agent_direction=Direction.UP,
            different_starting_cell=(1, 4),
        )

    def test_agent_returns_correct_state_facing_up_left_of_right_wall_two_below_corners(
        self,
    ):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.WALL,
                RelativePosition.RIGHT,
            ],
            obj_description="right wall",
            agent_direction=Direction.UP,
            different_starting_cell=(2, 4),
        )

    def test_agent_returns_correct_state_facing_down_left_of_right_wall(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.WALL,
                RelativePosition.LEFT,
            ],
            obj_description="right wall",
            agent_direction=Direction.DOWN,
            different_starting_cell=(1, 4),
        )

    def test_agent_returns_correct_state_facing_empty_tiles(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.NONE,
                RelativePosition.FRONT,
            ],
            obj_description="None",
            agent_direction=Direction.UP,
            different_starting_cell=(2, 2),
        )

    def test_agent_returns_correct_state_when_one_tile_below_and_facing_top_wall(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.WALL,
                RelativePosition.FRONT,
            ],
            obj_description="top wall",
            agent_direction=Direction.UP,
            different_starting_cell=(1, 1),
        )

    def test_agent_returns_correct_state_when_one_tile_left_and_facing_right_wall(self):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.WALL,
                RelativePosition.FRONT,
            ],
            obj_description="right wall",
            agent_direction=Direction.RIGHT,
            different_starting_cell=(1, 3),
        )

    def test_agent_returns_correct_state_when_one_tile_right_and_facing_away_right_wall(
        self,
    ):
        self.state_tester(
            correct_navigation_states=[
                ObjectType.NONE,
                RelativePosition.FRONT,
            ],
            obj_description="none",
            agent_direction=Direction.LEFT,
            different_starting_cell=(1, 3),
        )

    def test_agent_returns_correct_state_facing_top_left_corner_with_agent_on_right(
        self,
    ):
        SwarmAgent(id=2, starting_cell=self.tiled_enviro[(0, 1)])

        self.state_tester(
            correct_navigation_states=[
                ObjectType.CORNER,
                RelativePosition.LEFT_FRONT,
            ],
            obj_description="top left corner",
            agent_direction=Direction.UP,
        )

    def test_agent_returns_correct_state_at_top_left_corner_facing_agent_on_right(
        self,
    ):
        SwarmAgent(id=2, starting_cell=self.tiled_enviro[(0, 1)])

        self.state_tester(
            correct_navigation_states=[
                ObjectType.AGENT,
                RelativePosition.FRONT,
            ],
            obj_description="agent",
            agent_direction=Direction.RIGHT,
        )

    def test_agent_returns_correct_state_at_top_left_corner_but_facing_away_from_agent(
        self,
    ):
        SwarmAgent(id=2, starting_cell=self.tiled_enviro[(0, 1)])

        self.state_tester(
            correct_navigation_states=[
                ObjectType.CORNER,
                RelativePosition.RIGHT_FRONT,
            ],
            obj_description="top left corner",
            agent_direction=Direction.LEFT,
        )

    def test_agent_returns_correct_state_facing_agent_on_an_empty_square(self):
        SwarmAgent(id=2, starting_cell=self.tiled_enviro[(0, 1)])

        self.state_tester(
            correct_navigation_states=[
                ObjectType.AGENT,
                RelativePosition.FRONT,
            ],
            obj_description="agent",
            agent_direction=Direction.UP,
            different_starting_cell=(1, 1),
        )

    def test_agent_returns_correct_state_facing_agent_on_left_wall_square(self):
        SwarmAgent(id=2, starting_cell=self.tiled_enviro[(1, 1)])

        self.state_tester(
            correct_navigation_states=[
                ObjectType.AGENT,
                RelativePosition.FRONT,
            ],
            obj_description="agent",
            agent_direction=Direction.RIGHT,
            different_starting_cell=(1, 0),
        )


if __name__ == "__main__":
    unittest.main()
