def calculate_optimal_number_of_steps_needed(
    tiled_environment_width: int,
    tiled_environment_height: int,
) -> int:
    """
    Calculates the optimal number of steps needed to complete the task.
    """

    return int(
        (
            (tiled_environment_height * tiled_environment_width)
            - 2
            + (2 * tiled_environment_width)
        )
    )
