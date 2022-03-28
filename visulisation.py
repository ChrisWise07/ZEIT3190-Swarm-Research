from typing import Tuple
import cv2
import numpy as np
from random import choice
from environment_agent_modules import (
    create_nonclustered_tile_grid,
    SwarmAgent,
    Direction,
)

TILE_SIZE = 25
MID_POINT_OFFSET = round(TILE_SIZE / 2)


def setup_tile_grid_for_display(tile_grid: np.ndarray) -> np.ndarray:
    display_grid = np.zeros(
        (
            (tile_grid.shape[0] * TILE_SIZE) + TILE_SIZE,
            tile_grid.shape[1] * TILE_SIZE,
            3,
        ),
        np.uint8,
    )

    for row in range(tile_grid.shape[0]):
        for col in range(tile_grid.shape[1]):
            if not (tile_grid[row, col]["colour"]):
                display_grid[
                    row * TILE_SIZE : row * TILE_SIZE + TILE_SIZE,
                    col * TILE_SIZE : col * TILE_SIZE + TILE_SIZE,
                ] = (
                    255,
                    255,
                    255,
                )
    display_grid[
        tile_grid.shape[0] * TILE_SIZE : tile_grid.shape[0] * TILE_SIZE + TILE_SIZE,
    ] = (245, 135, 66)

    return display_grid


def map_swarm_agent_direction_position_to_display_tile_grid_arrow_coordinates(
    agent_direction: int, agent_position: Tuple[int, int]
) -> Tuple[int, int]:
    return {
        Direction.UP.value: lambda row, column: (
            (column - MID_POINT_OFFSET, row),
            (column - MID_POINT_OFFSET, row - TILE_SIZE),
        ),
        Direction.RIGHT.value: lambda row, column: (
            (column - TILE_SIZE, row - MID_POINT_OFFSET),
            (column, row - MID_POINT_OFFSET),
        ),
        Direction.DOWN.value: lambda row, column: (
            (column - MID_POINT_OFFSET, row - TILE_SIZE),
            (column - MID_POINT_OFFSET, row),
        ),
        Direction.LEFT.value: lambda row, column: (
            (column, row - MID_POINT_OFFSET),
            (column - TILE_SIZE, row - MID_POINT_OFFSET),
        ),
    }[agent_direction](*agent_position)


def draw_swarm_agent_on_tile_grid(
    display_tile_grid: np.ndarray,
    swarm_agent: SwarmAgent,
) -> np.ndarray:
    (
        start_point,
        end_point,
    ) = map_swarm_agent_direction_position_to_display_tile_grid_arrow_coordinates(
        agent_direction=swarm_agent.current_direction_facing,
        agent_position=[(value + 1) * TILE_SIZE for value in swarm_agent.current_cell],
    )

    return cv2.arrowedLine(
        display_tile_grid,
        start_point,
        end_point,
        (0, 0, 255),
        1,
        tipLength=0.4,
    )


def move_and_show_window(winname: str, img: np.ndarray, x: int, y: int):
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, x, y)
    cv2.imshow(winname, img)
    cv2.waitKey(0)


def main() -> None:
    tile_grid = create_nonclustered_tile_grid(20, 20)
    swarm_agent = SwarmAgent(id=1, starting_cell=tile_grid[19, 19])

    for i in range(10):
        img = draw_swarm_agent_on_tile_grid(
            display_tile_grid=setup_tile_grid_for_display(tile_grid),
            swarm_agent=swarm_agent,
        )

        obv, reward = (
            np.array(swarm_agent.get_navigation_states(tile_grid)),
            swarm_agent.return_navigation_reward(),
        )

        cv2.putText(
            img,
            f"Observation is {obv} and reward is {reward}",
            (0, img.shape[0] - round(MID_POINT_OFFSET / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        move_and_show_window(winname="Tile Grid", img=img, x=600, y=200)

        swarm_agent.perform_navigation_action(
            action=choice(range(0, 3)), tile_grid=tile_grid
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
