# Task 7 - Overtake Strategy / Planning

## Purpose

Task 7 plans short-horizon ego manoeuvres that avoid predicted opponents, decide when an overtake is allowed, and explain the chosen racing strategy.

## Inputs Used

- Task 3 `tracks.csv`: opponent state, velocity, confidence, and track IDs.
- Task 5 `predictions.csv`: future opponent trajectories when source timestamps match.
- Task 6 `behaviour_predictions.csv`: behaviour labels and confidence for behaviour-aware margins.
- Task 4 `track_boundaries.geojson`: loaded for map context; local planning uses an ego-frame corridor inferred from tracks because Task 3 outputs are ego-frame.
- `data/ego.mat` when available: ego speed estimate only; planning coordinates remain ego-frame.

## Planner Architecture

The innovation is a hybrid architecture: a deterministic safety-constrained receding-horizon planner generates candidate manoeuvres, while a lightweight learned risk model ranks them using features from tracking, mapping, trajectory prediction, and behaviour prediction. This gives data-driven adaptation without sacrificing safety or explainability.

This is not an end-to-end black-box planner. The learned model never overrides hard constraints; unsafe candidates are rejected before selection unless every candidate is unsafe, in which case the planner falls back to emergency braking or maximum-clearance evasive behaviour.

## Candidate Trajectories

At each timestamp the planner samples `keep_racing_line`, `follow_opponent`, `brake`, `evasive_left`, `evasive_right`, `overtake_left`, `overtake_right`, `abort_overtake`, and `emergency_brake`. Candidates use constant acceleration plus smooth lateral offset profiles over the receding horizon.

## Risk And Cost Function

- Collision risk from minimum distance to predicted opponent paths.
- Boundary risk from distance to the inferred racing corridor.
- Racing-line cost from lateral deviation.
- Progress reward for forward motion.
- Clean-overtake bonus when a side pass is already safe, improves clearance, and preserves progress.
- Smoothness cost from acceleration, lateral acceleration, and yaw-rate proxies.
- Behaviour-aware penalties for braking, overtaking attempts, recovery, or unknown behaviour.

## Behaviour-Aware Logic

- Braking opponents increase front-collision cost and favor braking/following unless a side path is clear.
- Overtaking opponents block the side they are predicted to move toward.
- Trajectory recovery and unknown labels increase uncertainty margins and block overtake attempts.
- Lane keeping uses normal margins.

## Learning-Based Innovation Module

The training table is built from every candidate trajectory. Pseudo-labels mark safe/unsafe, deterministic selected/not selected, low/medium/high risk, and a normalized risk score. A RandomForestRegressor is trained when enough data and scikit-learn are available. If training is unavailable, the deterministic planner is used alone and the fallback reason is written to `learning_model_summary.csv`.

## Outputs

- `planning_decisions.csv`
- `planned_trajectories.csv`
- `risk_assessment.csv`
- `planning_summary.csv`
- `strategy_timeline.png`
- `planning_bev.png`
- `overtake_scenarios.png`
- `learned_risk_training_data.csv`
- `learned_risk_scores.csv`
- `learning_model_summary.csv`
- `learning_feature_importance.png` when feature importance is available.

## Limitations

- Task 3 and Task 5 are ego-frame planning inputs, while Task 4 boundaries are world-frame. The planner therefore uses a conservative local corridor and draws boundaries as context rather than hard geometry.
- Pseudo-label learning is useful for strategy ranking, but it is not a substitute for labelled planning data.
- Candidate generation is deliberately lightweight and does not solve a full nonlinear optimal-control problem.

## How To Run

```bash
python src/task7_overtake_planning.py --tracks outputs/task3/tracks.csv --predictions outputs/task5/predictions.csv --behaviour outputs/task6/behaviour_predictions.csv --map-dir outputs/task4 --output-dir outputs/task7
```

The default command also works:

```bash
python src/task7_overtake_planning.py
```

## Configuration Snapshot

- Horizon: 2.00 s
- Step: 0.25 s
- Minimum safe distance: 4.50 m
- Track half-width: 9.00 m before inference

## Run Notes

- tracks: loaded 2752 rows from outputs\task3\tracks.csv
- predictions: loaded 9000 rows from outputs\task5\predictions.csv
- predictions: Task 5 future positions used where source timestamps match
- behaviour: loaded 2752 rows from outputs\task6\behaviour_predictions.csv
- map: loaded 12 GeoJSON boundary polylines for context
- corridor: half_width=9.00 m (inferred from Task 3 lateral track spread with Task 4 boundaries for visual context)
- ego: loaded ego.mat velocity log with 45131 samples

## Planning Summary

| metric                        | strategy         | value              | percentage         | notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------- | ---------------- | ------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| number_of_planning_frames     |                  | 197                |                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| mean_safety_margin            |                  | 4.147662712435047  |                    | m                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| minimum_safety_margin         |                  | -4.910300743460708 |                    | m                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| number_of_unsafe_frames       |                  | 19                 |                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| number_of_overtaking_attempts |                  | 21                 |                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| number_of_aborted_overtakes   |                  | 4                  |                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| number_of_emergency_brakes    |                  | 31                 |                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| learning_based_model_used     |                  | True               |                    | RandomForestRegressor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| notes_about_missing_inputs    |                  |                    |                    | tracks: loaded 2752 rows from outputs\task3\tracks.csv \| predictions: loaded 9000 rows from outputs\task5\predictions.csv \| predictions: Task 5 future positions used where source timestamps match \| behaviour: loaded 2752 rows from outputs\task6\behaviour_predictions.csv \| map: loaded 12 GeoJSON boundary polylines for context \| corridor: half_width=9.00 m (inferred from Task 3 lateral track spread with Task 4 boundaries for visual context) \| ego: loaded ego.mat velocity log with 45131 samples |
| selected_strategy_count       | follow_opponent  | 76                 | 38.578680203045685 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| selected_strategy_count       | keep_racing_line | 41                 | 20.81218274111675  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| selected_strategy_count       | emergency_brake  | 31                 | 15.736040609137056 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| selected_strategy_count       | overtake_right   | 14                 | 7.106598984771574  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| selected_strategy_count       | evasive_right    | 13                 | 6.598984771573604  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| selected_strategy_count       | overtake_left    | 7                  | 3.553299492385787  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| selected_strategy_count       | brake            | 6                  | 3.045685279187817  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| selected_strategy_count       | evasive_left     | 5                  | 2.5380710659898478 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| selected_strategy_count       | abort_overtake   | 4                  | 2.030456852791878  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| unsafe_candidate_fraction     |                  | 0.3745064861816131 |                    | candidate-level                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

## Learning Model Summary

| model_type            | number_of_training_samples | number_of_features | pseudo_label_type                   | train_score | validation_score | fallback_reason | top_features                                                                                                                                                                                                                     |
| --------------------- | -------------------------- | ------------------ | ----------------------------------- | ----------- | ---------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RandomForestRegressor | 1773                       | 15                 | normalized deterministic risk score | 0.993       | 0.987            |                 | collision_risk_score:0.853; progress_reward:0.061; behaviour_uncertainty_score:0.059; min_predicted_distance:0.011; racing_line_cost:0.007; opponent_behaviour_code:0.006; smoothness_cost:0.002; time_to_closest_approach:0.001 |
