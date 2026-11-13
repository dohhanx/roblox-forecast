#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 over the scored window.

One forecast is graded at every hour, so the file is the mixture of the
hourly distributions rather than the chart at a single horizon. Each hour
mixes three pieces:

* persistence: the latest rank, with the #1 seat pulled toward #2 because
  that seat turns over faster than the rest of the top five;
* diffusion: a shrunk rank velocity whose width grows and then saturates;
* a surge: games whose players are growing far faster than a time-of-day
  swing, aimed at the top when their rank still lags their player count.

Probability that does not belong to any game in the history is left unnamed.
Every game on the latest chart is named, so that leftover is shared by games
that are not in the history. A game with a known publish time is removed once
it is older than 28 days.

Reads $MAKEFUN_INPUT_DIR or /input. Writes $MAKEFUN_OUTPUT_DIR or /output.
Python 3.13, standard library only. Deterministic.
"""

import json
import math
import os
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Persistence half-life of the "chart has not moved" component, in hours.
PERSIST_TAU = 30.0
PERSIST_SIGMA = 1.25
# Rank-velocity memory. The slope is shrunk toward zero and the integrated
# drift saturates, because consecutive rank changes only weakly persist.
VEL_HALF_LIFE = 16.0
VEL_SHRINK = 3.5
VEL_TAU = 18.0
# Diffusion width grows quickly, then saturates. Late hours stay uncertain
# without flattening the whole chart into one uniform.
SIGMA0 = 0.8
SIGMA_MAX = 4.5
SIGMA_TAU = 26.0
# Surge component. One-step player counts swing with the hour of the day, so
# only a ~24 hour change, or a short move past the extreme tail, counts.
GROWTH_LONG_BASE = 0.012
GROWTH_LONG_WEIGHT = 1.8
GROWTH_SHORT_BASE = 0.10
GROWTH_SHORT_WEIGHT = 1.5
ROCKET_SCALE = 3.4
ROCKET_CAP = 0.72
ROCKET_GAP_DIV = 28.0
ROCKET_SIGMA0 = 4.2
# Share of each place left for games that are not on the chart yet.
NEW_BASE = 0.02
NEW_SLOPE = 0.0016
NEW_POWER = 1.35
NEW_CAP = 0.34
# Games that have sat on the chart for days without climbing drift downward.
STALE_HOURS = 140.0
STALE_DRIFT = 0.05
# Up-and-Coming only lists experiences published in the last 28 days.
MAX_AGE_HOURS = 28.0 * 24.0
PLACES = 10
MIN_P = 1e-4
MAX_ROWS = 100


def parse_hours(stamp):
    """ISO-8601 UTC timestamp as hours since the Unix epoch."""
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() / 3600.0


def load_inputs():
    """task.json plus every snapshot in history.ndjson, oldest first."""
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as handle:
        task = json.load(handle)
    snaps = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    if not snaps:
        raise SystemExit("history.ndjson is empty")
    return task, snaps


def weighted_slope(points, half_life):
    """Exponentially weighted slope of y on t. Returns (slope, weight sum)."""
    if len(points) < 2:
        return 0.0, 0.0
    t_last = points[-1][0]
    weight_sum = 0.0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    for stamp, value in points:
        weight = 0.5 ** ((t_last - stamp) / half_life)
        weight_sum += weight
        sum_x += weight * stamp
        sum_y += weight * value
        sum_xx += weight * stamp * stamp
        sum_xy += weight * stamp * value
    variance = sum_xx - sum_x * sum_x / weight_sum
    if variance < 1e-6:
        return 0.0, weight_sum
    covariance = sum_xy - sum_x * sum_y / weight_sum
    return covariance / variance, weight_sum


def _detail_map(snap):
    """universe id -> details dict, for snapshots that carried games/v1."""
    details = snap.get("details") or {}
    if not isinstance(details, dict):
        return {}
    mapped = {}
    for key, detail in details.items():
        if not isinstance(detail, dict):
            continue
        try:
            mapped[int(key)] = detail
        except (TypeError, ValueError):
            continue
    return mapped


def _optional_float(detail, key):
    value = detail.get(key) if isinstance(detail, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def series_and_created(snaps):
    """Per-game history plus publish times.

    Each observation is (time, rank, players, player-rank, visits). Visits
    stay 0 unless that snapshot's details name them. `created` is set only
    when a snapshot carries games/v1 details: the practice archive has none,
    and the final hourly recorder does.
    """
    times = [parse_hours(snap["at"]) for snap in snaps]
    series = {}
    created = {}
    for snap, stamp in zip(snaps, times):
        rows = snap.get("up_and_coming") or []
        details = _detail_map(snap)
        by_players = sorted(range(len(rows)), key=lambda i: -(rows[i].get("playing") or 0))
        player_rank = {}
        for index, row_index in enumerate(by_players):
            player_rank[rows[row_index]["universe_id"]] = index + 1
        for row in rows:
            uid = row["universe_id"]
            detail = details.get(uid, {})
            series.setdefault(uid, []).append(
                (
                    stamp,
                    float(row["rank"]),
                    float(row.get("playing") or 0.0),
                    float(player_rank[uid]),
                    _optional_float(detail, "visits"),
                )
            )
            if uid not in created and detail.get("created"):
                try:
                    created[uid] = parse_hours(detail["created"])
                except (TypeError, ValueError):
                    continue
    return times, series, created


def _log_change(obs, index):
    """Log growth per hour of observation field `index`, or None if unknown.

    The long window is the observation nearest 24 hours back, kept only when
    it actually falls between 8 and 40 hours ago. Cumulative visits use the
    same window; concurrent players also report the last step.
    """
    last = obs[-1]
    if last[index] <= 0.0:
        return None, 0.0
    log_short = 0.0
    if len(obs) >= 2 and obs[-2][index] > 0.0:
        delta = last[0] - obs[-2][0]
        if delta > 0.25:
            log_short = (math.log(last[index] + 1.0) - math.log(obs[-2][index] + 1.0)) / delta
    best = None
    target = last[0] - 24.0
    for obs_row in obs[:-1]:
        if obs_row[index] <= 0.0:
            continue
        if best is None or abs(obs_row[0] - target) < abs(best[0] - target):
            best = obs_row
    log_long = None
    if best is not None and 8.0 < (last[0] - best[0]) < 40.0:
        log_long = (math.log(last[index] + 1.0) - math.log(best[index] + 1.0)) / (last[0] - best[0])
    return log_long, log_short


def _log_player_growth(obs):
    """Player log-growth. Visits, when the recorder sent them, can only raise it.

    Visits are cumulative, so a 24-hour visit change is not a time-of-day
    swing. The practice archive sends no visits and this leaves the player
    signal untouched.
    """
    visit_long, _visit_short = _log_change(obs, 4)
    player_long, player_short = _log_change(obs, 2)
    log_long = 0.0 if player_long is None else player_long
    if visit_long is not None:
        log_long = max(log_long, visit_long)
    return player_short, log_long


def game_features(series, now):
    """Features for games still on the latest chart."""
    features = {}
    for uid, obs in series.items():
        last = obs[-1]
        if abs(last[0] - now) > 1e-3:
            continue
        slope, weight = weighted_slope(
            [(stamp, rank) for stamp, rank, _players, _player_rank, _visits in obs],
            VEL_HALF_LIFE,
        )
        shrink = weight / (weight + VEL_SHRINK)
        log_short, log_long = _log_player_growth(obs)
        features[uid] = {
            "rank": last[1],
            "vel": slope * shrink,
            "log_short": log_short,
            "log_long": log_long,
            "gap": last[1] - last[3],
            "hours_on": now - obs[0][0],
            "n": len(obs),
        }
    return features


def gaussian(place, mean, sigma):
    """Density of a normal rank at an integer place. Far tails are zero."""
    z = (place - mean) / sigma
    if abs(z) > 8.0:
        return 0.0
    return math.exp(-0.5 * z * z) / sigma


def survival(uid, created, now, horizon):
    """1 while the game can still be listed `horizon` hours after `now`."""
    born = created.get(uid)
    if born is None:
        return 1.0
    remaining = born + MAX_AGE_HOURS - now
    if horizon > remaining:
        return 0.0
    if horizon > remaining - 12.0:
        return 0.4
    return 1.0


def rocket_weight(feat):
    """Mixture weight of a jump toward the top, from shrunk player growth.

    A game already ranked far above its player count is not treated as a
    climber: the surge there is popularity the chart has already priced in.
    """
    signal = max(feat["log_long"] - GROWTH_LONG_BASE, 0.0) * GROWTH_LONG_WEIGHT
    if feat["log_short"] > GROWTH_SHORT_BASE:
        signal = max(signal, (feat["log_short"] - GROWTH_SHORT_BASE) * GROWTH_SHORT_WEIGHT)
    weight = min(ROCKET_CAP, signal * ROCKET_SCALE)
    if feat["gap"] > 8.0 and feat["rank"] >= 8.0:
        weight = min(ROCKET_CAP, weight * (1.0 + min(feat["gap"], 45.0) / ROCKET_GAP_DIV))
    if feat["gap"] < -15.0:
        weight *= 0.2
    if feat["rank"] <= 4.0:
        weight *= 0.3
    return weight


def rocket_target(feat, weight):
    """Rank the surge is heading toward. Underranked growers aim at the top."""
    if weight >= 0.15 and feat["gap"] > 8.0:
        return max(1.5, min(5.5, feat["rank"] * 0.13))
    if weight >= 0.12:
        return max(3.0, min(12.0, feat["rank"] * 0.15))
    return max(1.0, feat["rank"] * 0.5)


def _volatility(observations):
    if observations <= 2:
        return 1.45
    if observations <= 4:
        return 1.2
    return 1.0


def _is_stale(feat):
    return (
        feat["hours_on"] > STALE_HOURS
        and feat["vel"] > -0.04
        and feat["log_long"] < 0.01
        and feat["log_short"] < 0.04
    )


def prepare_candidates(features):
    """Latest-chart games with surge weight, target rank, staleness, noise."""
    prepared = []
    for uid in features:
        feat = features[uid]
        weight = rocket_weight(feat)
        prepared.append(
            (uid, feat, weight, rocket_target(feat, weight), _is_stale(feat), _volatility(feat["n"]))
        )
    return prepared


def _persist_center(rank):
    """Sticky rank. The #1 seat turns over faster than ranks 3-5, so the
    leader and the game under it are centered between those two seats.
    """
    if rank <= 1.0:
        return 1.8
    if rank <= 2.0:
        return 1.45
    return rank


def densities_at_horizon(prepared, created, now, horizon, places):
    """Place densities for one hour, and the newcomer share held out of them.

    A surge is mixed in immediately. Waiting for persistence to fade left
    those games with almost no mass across the 72-hour average. A positive
    rank velocity (the place number getting worse) is dropped for a surge,
    because that lag is the thing the player jump is meant to correct.
    """
    sigma_h = SIGMA0 + SIGMA_MAX * (1.0 - math.exp(-horizon / SIGMA_TAU))
    persist = math.exp(-horizon / PERSIST_TAU)
    newcomer = min(NEW_CAP, NEW_BASE + NEW_SLOPE * (horizon ** NEW_POWER))
    densities = []
    for uid, feat, weight, rocket_mean, stale, volatility in prepared:
        alive = survival(uid, created, now, horizon)
        if alive <= 0.0:
            densities.append(None)
            continue
        drift = feat["vel"] * VEL_TAU * (1.0 - math.exp(-horizon / VEL_TAU))
        if weight > 0.2 and drift > 0.0:
            drift = 0.0
        mean = feat["rank"] + drift
        if stale:
            mean += STALE_DRIFT * horizon
        sigma = max(0.7, sigma_h * volatility)
        rocket_sigma = max(3.0, ROCKET_SIGMA0 + 0.035 * horizon)
        persist_w = persist * (1.0 - 0.9 * weight)
        row = [0.0] * places
        center = _persist_center(feat["rank"])
        for place in range(1, places + 1):
            density = persist_w * gaussian(place, center, PERSIST_SIGMA)
            density += (1.0 - persist_w) * (1.0 - weight) * gaussian(place, mean, sigma)
            if weight > 0.02:
                density += (1.0 - persist_w) * weight * gaussian(place, rocket_mean, rocket_sigma)
            row[place - 1] = alive * density
        densities.append(row)
    return densities, newcomer


def forecast(task, snaps):
    """Horizon-averaged place probabilities and an ordered top-10 guess."""
    times, series, created = series_and_created(snaps)
    now = times[-1]
    features = game_features(series, now)
    prepared = prepare_candidates(features)

    win_from = parse_hours(task["window"]["from"])
    win_until = parse_hours(task["window"]["until"])
    slots = int(task["window"]["slots"])
    places = int(task.get("places", PLACES))
    slot_hours = (win_until - win_from) / slots
    horizons = [max(0.3, win_from + (index + 0.5) * slot_hours - now) for index in range(slots)]

    accumulated = [[0.0] * places for _ in prepared]
    for horizon in horizons:
        densities, newcomer = densities_at_horizon(prepared, created, now, horizon, places)
        for place_index in range(places):
            total = 0.0
            for row in densities:
                if row is not None:
                    total += row[place_index]
            if total <= 0.0:
                continue
            scale = (1.0 - newcomer) / total
            for index, row in enumerate(densities):
                if row is not None:
                    accumulated[index][place_index] += row[place_index] * scale

    n_horizons = float(len(horizons))
    by_place = [[] for _ in range(places)]
    for index, (uid, _feat, _weight, _mean, _stale, _vol) in enumerate(prepared):
        for place_index in range(places):
            probability = accumulated[index][place_index] / n_horizons
            if probability > 0.0:
                by_place[place_index].append((uid, probability))

    probabilities = _name_every_chart_game(by_place, prepared)
    top = _top_list(probabilities, features, snaps, places)
    return probabilities, top, task.get("output", "forecast.json")


def _name_every_chart_game(by_place, prepared):
    """Emit up to 100 rows per place, naming every current-chart game.

    Dust below the log-score floor is dropped before the fill. Leaving it in
    counted the game as named, then rounding deleted the row, and the game
    went back into the residual pool and thinned the share left for newcomers.
    """
    probabilities = []
    for place_index, rows in enumerate(by_place):
        rows = [(uid, probability) for uid, probability in rows if probability >= MIN_P]
        rows.sort(key=lambda item: (-item[1], item[0]))
        mass = sum(probability for _uid, probability in rows)
        if mass > 0.97:
            scale = 0.97 / mass
            rows = [(uid, probability * scale) for uid, probability in rows]
            mass = 0.97
        named = {uid for uid, _probability in rows}
        for uid, _feat, _weight, _mean, _stale, _vol in prepared:
            if len(rows) >= MAX_ROWS:
                break
            if uid in named:
                continue
            if mass + MIN_P > 0.98:
                break
            rows.append((uid, MIN_P))
            mass += MIN_P
            named.add(uid)
        place = place_index + 1
        for uid, probability in rows:
            if probability >= MIN_P:
                probabilities.append((uid, place, probability))
    return probabilities


def _top_list(probabilities, features, snaps, places):
    """Ten ids ordered by conditional expected place, mass as the tie-break."""
    mass = {}
    expected = {}
    for uid, place, probability in probabilities:
        mass[uid] = mass.get(uid, 0.0) + probability
        expected[uid] = expected.get(uid, 0.0) + place * probability

    def order_key(uid):
        total = mass.get(uid, 0.0)
        if total <= 1e-6:
            return (50.0, 0.0, uid)
        return (expected[uid] / total, -total, uid)

    top = sorted(features, key=order_key)[:places]
    if len(top) < places:
        chart = sorted(snaps[-1]["up_and_coming"], key=lambda row: row["rank"])
        for row in chart:
            if row["universe_id"] not in top:
                top.append(row["universe_id"])
            if len(top) == places:
                break
    return top


def round_probabilities(probabilities):
    """Six-decimal probabilities, at most 100 per place, each place summing to <= 1."""
    by_place = {}
    for uid, place, probability in probabilities:
        by_place.setdefault(place, []).append([uid, round(float(probability), 6)])
    rows = []
    for place in sorted(by_place):
        items = [item for item in by_place[place] if item[1] >= MIN_P]
        items.sort(key=lambda item: (-item[1], item[0]))
        items = items[:MAX_ROWS]
        total = sum(item[1] for item in items)
        if total > 1.0:
            overflow = total - 1.0
            items[0][1] = round(items[0][1] - overflow, 6)
            if items[0][1] < MIN_P:
                items[0][1] = MIN_P
        for uid, probability in items:
            if probability >= MIN_P:
                rows.append({"universe_id": int(uid), "place": int(place), "p": probability})
    rows.sort(key=lambda row: (row["place"], -row["p"], row["universe_id"]))
    return rows


def main():
    task, snaps = load_inputs()
    probabilities, top10, output_name = forecast(task, snaps)
    document = {
        "probabilities": round_probabilities(probabilities),
        "top10": [int(uid) for uid in top10],
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, output_name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    main()
