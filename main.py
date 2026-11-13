#!/usr/bin/env python3
"""Forecast Roblox Up-and-Coming top-10 places over the scored window.

The chart is sticky for a few hours and then diffuses, while a few games with
extreme player growth jump toward the top and brand-new games take a growing
share of each place. One forecast is scored at every hour, so the file is the
mixture of those hourly distributions, not the chart at a single horizon.

Reads $MAKEFUN_INPUT_DIR or /input. Writes $MAKEFUN_OUTPUT_DIR or /output.
Standard library only. Deterministic.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Persistence half-life of the "chart has not moved" component, in hours.
PERSIST_TAU = 30.0
PERSIST_SIGMA = 1.25
# Rank-velocity memory. Slope is shrunk toward 0; integrated drift saturates.
VEL_HALF_LIFE = 16.0
VEL_SHRINK = 3.5
VEL_TAU = 18.0
# Diffusion width grows quickly, then saturates: late hours stay uncertain
# but do not flatten the whole chart into one uniform.
SIGMA0 = 0.8
SIGMA_MAX = 4.5
SIGMA_TAU = 26.0
# Rocket component for games whose players are growing far faster than a
# diurnal swing, and whose rank is worse than their player count.
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
# A game published more than 28 days ago cannot remain on this chart.
MAX_AGE_HOURS = 28.0 * 24.0
PLACES = 10
MIN_P = 1e-4


def parse_hours(stamp):
    """ISO-8601 UTC timestamp -> hours since the Unix epoch."""
    from datetime import datetime

    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() / 3600.0


def load_inputs():
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
    sw = sx = sy = sxx = sxy = 0.0
    for t, y in points:
        weight = 0.5 ** ((t_last - t) / half_life)
        sw += weight
        sx += weight * t
        sy += weight * y
        sxx += weight * t * t
        sxy += weight * t * y
    var = sxx - sx * sx / sw
    if var < 1e-6:
        return 0.0, sw
    cov = sxy - sx * sy / sw
    return cov / var, sw


def series_and_created(snaps):
    """Per-game (time, rank, players, player-rank) history, plus created times."""
    times = [parse_hours(snap["at"]) for snap in snaps]
    series = {}
    for snap, stamp in zip(snaps, times):
        rows = snap.get("up_and_coming") or []
        by_players = sorted(range(len(rows)), key=lambda i: -(rows[i].get("playing") or 0))
        player_rank = {}
        for index, row_index in enumerate(by_players):
            player_rank[rows[row_index]["universe_id"]] = index + 1
        for row in rows:
            series.setdefault(row["universe_id"], []).append(
                (
                    stamp,
                    float(row["rank"]),
                    float(row.get("playing") or 0.0),
                    float(player_rank[row["universe_id"]]),
                )
            )
    created = {}
    for snap in snaps:
        details = snap.get("details") or {}
        if not isinstance(details, dict):
            continue
        for key, detail in details.items():
            if not isinstance(detail, dict) or not detail.get("created"):
                continue
            try:
                created[int(key)] = parse_hours(detail["created"])
            except (TypeError, ValueError):
                continue
    return times, series, created


def game_features(series, now):
    """Features for games on the latest chart."""
    features = {}
    for uid, obs in series.items():
        last = obs[-1]
        if abs(last[0] - now) > 1e-3:
            continue
        slope, weight = weighted_slope([(t, rank) for t, rank, _players, _prank in obs], VEL_HALF_LIFE)
        shrink = weight / (weight + VEL_SHRINK)
        log_short = 0.0
        if len(obs) >= 2 and obs[-2][2] > 0.0 and last[2] > 0.0:
            dt = last[0] - obs[-2][0]
            if dt > 0.25:
                log_short = (math.log(last[2] + 1.0) - math.log(obs[-2][2] + 1.0)) / dt
        log_long = 0.0
        best = None
        target = last[0] - 24.0
        for stamp, _rank, players, _prank in obs[:-1]:
            if players <= 0.0:
                continue
            if best is None or abs(stamp - target) < abs(best[0] - target):
                best = (stamp, players)
        if best and last[2] > 0.0 and 8.0 < (last[0] - best[0]) < 40.0:
            log_long = (math.log(last[2] + 1.0) - math.log(best[1] + 1.0)) / (last[0] - best[0])
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
    z = (place - mean) / sigma
    if abs(z) > 8.0:
        return 0.0
    return math.exp(-0.5 * z * z) / sigma


def survival(uid, created, now, horizon):
    """1 if the game can still be on the chart `horizon` hours after `now`."""
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
    """Probability of a jump toward the top, from shrunk player-growth.

    One-step player counts swing with the time of day, so only a 24-hour
    change or a short move past the extreme tail counts. A game already
    ranked far above its player count is not treated as a climber.
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


def forecast(task, snaps):
    times, series, created = series_and_created(snaps)
    now = times[-1]
    features = game_features(series, now)
    candidates = list(features)
    prepared = []
    for uid in candidates:
        feat = features[uid]
        weight = rocket_weight(feat)
        rocket_mean = rocket_target(feat, weight)
        stale = (
            feat["hours_on"] > STALE_HOURS
            and feat["vel"] > -0.04
            and feat["log_long"] < 0.01
            and feat["log_short"] < 0.04
        )
        if feat["n"] <= 2:
            volatility = 1.45
        elif feat["n"] <= 4:
            volatility = 1.2
        else:
            volatility = 1.0
        prepared.append((uid, feat, weight, rocket_mean, stale, volatility))

    win_from = parse_hours(task["window"]["from"])
    win_until = parse_hours(task["window"]["until"])
    slots = int(task["window"]["slots"])
    places = int(task.get("places", PLACES))
    slot_hours = (win_until - win_from) / slots
    horizons = [max(0.3, win_from + (index + 0.5) * slot_hours - now) for index in range(slots)]

    accumulated = [[0.0] * places for _ in prepared]
    for horizon in horizons:
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
            # A lagging rank (players up, place down) must not drag a rocket
            # the wrong way. Keep only upward velocity for those games.
            if weight > 0.2 and drift > 0.0:
                drift = 0.0
            mean = feat["rank"] + drift
            if stale:
                mean += STALE_DRIFT * horizon
            # The #1 slot turns over much faster than the rest of the top 5,
            # so persistence for the leader is centered between #1 and #2.
            persist_center = feat["rank"]
            if feat["rank"] <= 1.0:
                persist_center = 1.8
            elif feat["rank"] <= 2.0:
                persist_center = 1.45
            sigma = max(0.7, sigma_h * volatility)
            rocket_sigma = max(3.0, ROCKET_SIGMA0 + 0.035 * horizon)
            # The jump is available immediately. Waiting for persistence to
            # fade kept surges at ~0.01 through the whole window.
            persist_w = persist * (1.0 - 0.9 * weight)
            row = [0.0] * places
            for place in range(1, places + 1):
                density = persist_w * gaussian(place, persist_center, PERSIST_SIGMA)
                density += (1.0 - persist_w) * (1.0 - weight) * gaussian(place, mean, sigma)
                if weight > 0.02:
                    density += (1.0 - persist_w) * weight * gaussian(place, rocket_mean, rocket_sigma)
                row[place - 1] = alive * density
            densities.append(row)
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

    # Name every current-chart game so leftover mass concentrates on games
    # that are not in the history (the newcomer share), up to 100 rows.
    probabilities = []
    for place_index, rows in enumerate(by_place):
        # Drop dust before the fill. A sub-floor row used to count as "named"
        # and then disappear at rounding, putting that game back in the
        # residual pool and thinning the share left for true newcomers.
        rows = [(uid, probability) for uid, probability in rows if probability >= MIN_P]
        rows.sort(key=lambda item: (-item[1], item[0]))
        mass = sum(probability for _uid, probability in rows)
        if mass > 0.97:
            scale = 0.97 / mass
            rows = [(uid, probability * scale) for uid, probability in rows]
            mass = 0.97
        named = {uid for uid, _probability in rows}
        tiny = MIN_P
        for uid, _feat, _weight, _mean, _stale, _vol in prepared:
            if len(rows) >= 100:
                break
            if uid in named:
                continue
            if mass + tiny > 0.98:
                break
            rows.append((uid, tiny))
            mass += tiny
            named.add(uid)
        place = place_index + 1
        for uid, probability in rows:
            if probability >= MIN_P:
                probabilities.append((uid, place, probability))

    mass = {}
    expected = {}
    for uid, place, probability in probabilities:
        mass[uid] = mass.get(uid, 0.0) + probability
        expected[uid] = expected.get(uid, 0.0) + place * probability

    def order_key(uid):
        total = mass.get(uid, 0.0)
        if total <= 1e-6:
            return (50.0, 0.0, uid)
        # Conditional expected place. Mass breaks ties toward games we
        # actually put probability on.
        return (expected[uid] / total, -total, uid)

    top = sorted(candidates, key=order_key)[:places]
    if len(top) < places:
        chart = sorted(snaps[-1]["up_and_coming"], key=lambda row: row["rank"])
        for row in chart:
            if row["universe_id"] not in top:
                top.append(row["universe_id"])
            if len(top) == places:
                break
    return probabilities, top, task.get("output", "forecast.json")


def round_probabilities(probabilities):
    """Six-decimal probabilities, at most 100 per place, each place summing to <= 1."""
    by_place = {}
    for uid, place, probability in probabilities:
        by_place.setdefault(place, []).append([uid, round(probability, 6)])
    rows = []
    for place in sorted(by_place):
        items = [item for item in by_place[place] if item[1] >= MIN_P]
        items.sort(key=lambda item: (-item[1], item[0]))
        items = items[:100]
        total = sum(item[1] for item in items)
        if total > 1.0:
            overflow = total - 1.0
            items.sort(key=lambda item: (-item[1], item[0]))
            items[0][1] = round(items[0][1] - overflow, 6)
            if items[0][1] < MIN_P:
                items[0][1] = MIN_P
        for uid, probability in items:
            if probability >= MIN_P:
                rows.append({"universe_id": uid, "place": place, "p": probability})
    rows.sort(key=lambda row: (row["place"], -row["p"], row["universe_id"]))
    return rows


def main():
    task, snaps = load_inputs()
    probabilities, top10, output_name = forecast(task, snaps)
    document = {
        "probabilities": round_probabilities(probabilities),
        "top10": top10,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, output_name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    main()
