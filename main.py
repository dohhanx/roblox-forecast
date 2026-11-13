# Up-and-Coming mixture forecaster.
# One table is scored on every hour of the window, so the program averages
# the hourly seat distributions. Each hour blends a sticky seat, a shrunk
# rank slope whose spread grows and then levels off, and a jump for titles
# whose audience is rising much faster than the time of day. Unnamed mass
# is what the history does not claim. A known publish instant drops a title
# once it is older than four weeks.
# Invoked as python3 main.py. Standard library only. No randomness.

import json as chartjson
import math as chartmath
import os as chartos
from datetime import datetime as ChartClock

CHART_IN = chartos.environ.get("MAKEFUN_INPUT_DIR", "/input")
CHART_OUT = chartos.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Sticky component: half-life and gaussian width, both in hours / seats.
STICK_HALFLIFE = 30.0
STICK_WIDTH = 1.25
# Slope of seat versus time. Prior strength shrinks a short sample toward 0.
SLOPE_HALFLIFE = 16.0
SLOPE_PRIOR = 3.5
SLOPE_MEMORY = 18.0
# Diffusive width starts narrow and saturates. Late hours stay unsure
# without washing the whole list into one flat distribution.
WIDTH_FLOOR = 0.8
WIDTH_SPAN = 4.5
WIDTH_HALFLIFE = 26.0
# Audience growth. A single step moves with the hour, so only a day-scale
# change, or a burst past the extreme tail, is allowed to launch a jump.
LONG_GROWTH_FLOOR = 0.012
LONG_GROWTH_GAIN = 1.8
BURST_GROWTH_FLOOR = 0.10
BURST_GROWTH_GAIN = 1.5
JUMP_GAIN = 3.4
JUMP_LIMIT = 0.72
GAP_DIVISOR = 28.0
JUMP_WIDTH0 = 4.2
# Share of a seat reserved for titles that have not appeared in the history.
FRESH_FLOOR = 0.02
FRESH_SLOPE = 0.0016
FRESH_POWER = 1.35
FRESH_LIMIT = 0.34
# Long-resident titles that are not climbing slide slowly down the list.
AGED_AFTER = 140.0
AGED_SLIDE = 0.05
LISTING_LIFE = 28.0 * 24.0
SEAT_COUNT = 10
FLOOR_P = 1e-4
ROW_CAP = 100
# Like-ratio near 0.90 is neutral. A large sample can shift the diffused seat.
NEUTRAL_LIKE = 0.90
LIKE_PULL = 8.0
LIKE_VOTES = 50.0


def as_hours(text):
    """UTC timestamp text, in hours since 1970-01-01."""
    cleaned = text.replace("Z", "+00:00")
    instant = ChartClock.fromisoformat(cleaned)
    return instant.timestamp() / 3600.0


def read_bundle():
    """task.json together with history.ndjson, oldest snapshot first."""
    task_path = chartos.path.join(CHART_IN, "task.json")
    history_path = chartos.path.join(CHART_IN, "history.ndjson")
    with open(task_path, encoding="utf-8") as task_file:
        task = chartjson.load(task_file)
    snapshots = []
    with open(history_path, encoding="utf-8") as history_file:
        for raw in history_file:
            raw = raw.strip()
            if raw:
                snapshots.append(chartjson.loads(raw))
    if len(snapshots) == 0:
        raise SystemExit("history.ndjson has no snapshots")
    return task, snapshots


def slope_fit(samples, halflife):
    """Weighted least-squares slope of value on time. Also returns weight mass."""
    if not samples or len(samples) == 1:
        nothing = (0.0, 0.0)
        return nothing
    newest = samples[-1][0]
    mass = 0.0
    mom_t = 0.0
    mom_y = 0.0
    mom_tt = 0.0
    mom_ty = 0.0
    for moment, value in samples:
        age = newest - moment
        decay = chartmath.pow(0.5, age / halflife)
        mass += decay
        mom_t += decay * moment
        mom_y += decay * value
        mom_tt += decay * moment * moment
        mom_ty += decay * moment * value
    spread = mom_tt - (mom_t * mom_t) / mass
    if spread <= 1e-6:
        flat = (0.0, mass)
        return flat
    co_move = mom_ty - (mom_t * mom_y) / mass
    rate = co_move / spread
    return (rate, mass)


def numeric(value):
    """JSON numbers count. Booleans are ints and do not."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def coerce_uid(token):
    try:
        return int(token), True
    except Exception:
        return 0, False


def detail_table(snapshot):
    """universe id to games/v1 object, for snapshots that carried details."""
    blob = snapshot.get("details") or {}
    found = {}
    if not isinstance(blob, dict):
        return found
    for key, body in blob.items():
        if isinstance(body, dict):
            parsed, ok = coerce_uid(key)
            if ok:
                found[parsed] = body
    return found


def detail_number(info, field):
    value = info.get(field) if isinstance(info, dict) else None
    if not numeric(value):
        missing = 0.0
        return missing
    return float(value)


def vote_counts(entry, info):
    """Up and down votes from the chart row, else from details."""
    up = entry.get("up_votes")
    down = entry.get("down_votes")
    if numeric(up) and numeric(down):
        return float(up), float(down)
    if isinstance(info, dict):
        up = info.get("up_votes")
        down = info.get("down_votes")
        if numeric(up) and numeric(down):
            return float(up), float(down)
    return None


def approval_shift(entry, info):
    """Seats to add to the diffused mean. Negative is a better place.

    No votes means no shift. The practice archive omits votes entirely.
    """
    pair = vote_counts(entry, info)
    if pair is None:
        neutral = 0.0
        return neutral
    up, down = pair
    votes = up + down
    if votes < LIKE_VOTES:
        neutral = 0.0
        return neutral
    ratio = up / votes
    return (NEUTRAL_LIKE - ratio) * LIKE_PULL


def build_tracks(snapshots):
    """Per-title history plus publish instants.

    A point is (when, seat, audience, audience-seat, visits, favorites, nudge).
    Visits, favorites and the nudge stay zero unless the snapshot names them.
    """
    moments = [as_hours(shot["at"]) for shot in snapshots]
    tracks = {}
    born = {}
    for shot, moment in zip(snapshots, moments):
        rows = shot.get("up_and_coming") or []
        info = detail_table(shot)
        order = sorted(range(len(rows)), key=lambda i: -(rows[i].get("playing") or 0))
        by_audience = {}
        for place, row_index in enumerate(order):
            by_audience[rows[row_index]["universe_id"]] = place + 1
        for entry in rows:
            uid = entry["universe_id"]
            extra = info.get(uid, {})
            audience = entry.get("playing")
            if not numeric(audience):
                audience = 0.0
            point = (
                moment,
                float(entry["rank"]),
                float(audience),
                float(by_audience[uid]),
                detail_number(extra, "visits"),
                detail_number(extra, "favorites"),
                approval_shift(entry, extra),
            )
            tracks.setdefault(uid, []).append(point)
            if uid not in born and extra.get("created"):
                try:
                    born[uid] = as_hours(extra["created"])
                except Exception:
                    born.pop(uid, None)
    return moments, tracks, born


def growth_pair(track, field):
    """(day-scale log growth or None, last-step log growth) for one field.

    The day-scale reading is the sample nearest a day earlier, and only if
    that sample sits between eight and forty hours back.
    """
    latest = track[-1]
    if latest[field] <= 0.0:
        unknown = (None, 0.0)
        return unknown
    step = 0.0
    if len(track) >= 2 and track[-2][field] > 0.0:
        gap = latest[0] - track[-2][0]
        if gap > 0.25:
            step = chartmath.log(latest[field] + 1.0) - chartmath.log(track[-2][field] + 1.0)
            step = step / gap
    aim = latest[0] - 24.0
    chosen = None
    for earlier in track[:-1]:
        if earlier[field] <= 0.0:
            pass
        elif chosen is None or abs(earlier[0] - aim) < abs(chosen[0] - aim):
            chosen = earlier
    day = None
    if chosen is not None:
        span = latest[0] - chosen[0]
        if span > 8.0 and span < 40.0:
            day = chartmath.log(latest[field] + 1.0) - chartmath.log(chosen[field] + 1.0)
            day = day / span
    return day, step


def audience_signal(track):
    """Player log-growth. Visits and favorites may only lift the day-scale rate."""
    day, step = growth_pair(track, 2)
    lifted = 0.0 if day is None else day
    for field in (4, 5):
        extra, _ignored = growth_pair(track, field)
        if extra is not None and extra > lifted:
            lifted = extra
    return step, lifted


def features_now(tracks, now):
    """Titles still present on the newest chart."""
    found = {}
    for uid, track in tracks.items():
        latest = track[-1]
        if abs(latest[0] - now) > 1e-3:
            pass
        else:
            pairs = [(point[0], point[1]) for point in track]
            rate, mass = slope_fit(pairs, SLOPE_HALFLIFE)
            shrink = mass / (mass + SLOPE_PRIOR)
            step, lifted = audience_signal(track)
            found[uid] = {
                "seat": latest[1],
                "pace": rate * shrink,
                "burst": step,
                "day": lifted,
                "lag": latest[1] - latest[3],
                "resident": now - track[0][0],
                "seen": len(track),
                "nudge": latest[6],
            }
    return found


def jump_mass(feat):
    """Weight of a leap toward the top, from shrunk audience growth.

    A title already seated far above its audience is not treated as a climber.
    """
    signal = max(feat["day"] - LONG_GROWTH_FLOOR, 0.0) * LONG_GROWTH_GAIN
    if feat["burst"] > BURST_GROWTH_FLOOR:
        burst = (feat["burst"] - BURST_GROWTH_FLOOR) * BURST_GROWTH_GAIN
        signal = max(signal, burst)
    weight = min(JUMP_LIMIT, signal * JUMP_GAIN)
    if feat["lag"] > 8.0 and feat["seat"] >= 8.0:
        boost = 1.0 + min(feat["lag"], 45.0) / GAP_DIVISOR
        weight = min(JUMP_LIMIT, weight * boost)
    if feat["lag"] < -15.0:
        weight = weight * 0.2
    if feat["seat"] <= 4.0:
        weight = weight * 0.3
    return weight


def jump_seat(feat, weight):
    """Seat the surge is aimed at. Underranked growers point at the top."""
    if weight >= 0.15 and feat["lag"] > 8.0:
        aimed = max(1.5, min(5.5, feat["seat"] * 0.13))
        return aimed
    if weight >= 0.12:
        aimed = max(3.0, min(12.0, feat["seat"] * 0.15))
        return aimed
    aimed = max(1.0, feat["seat"] * 0.5)
    return aimed


def noise_scale(seen):
    if seen <= 2:
        broad = 1.45
        return broad
    if seen <= 4:
        mid = 1.2
        return mid
    calm = 1.0
    return calm


def sitting_too_long(feat):
    old = feat["resident"] > AGED_AFTER
    flat_pace = feat["pace"] > -0.04
    quiet_day = feat["day"] < 0.01
    quiet_step = feat["burst"] < 0.04
    return old and flat_pace and quiet_day and quiet_step


def pack_games(features):
    """Newest-chart titles with jump weight, aim, staleness and noise."""
    packed = []
    for uid in features:
        feat = features[uid]
        weight = jump_mass(feat)
        packed.append(
            (uid, feat, weight, jump_seat(feat, weight), sitting_too_long(feat), noise_scale(feat["seen"]))
        )
    return packed


def sticky_seat(seat):
    """Sticky center. Seat 1 turns over faster than seats 3-5."""
    if seat <= 1.0:
        between = 1.8
        return between
    if seat <= 2.0:
        between = 1.45
        return between
    return seat


def still_listed(uid, born, now, ahead):
    """1 while the title can remain listed `ahead` hours after `now`."""
    opened = born.get(uid)
    if opened is None:
        full = 1.0
        return full
    left = opened + LISTING_LIFE - now
    if ahead > left:
        gone = 0.0
        return gone
    if ahead > left - 12.0:
        fading = 0.4
        return fading
    full = 1.0
    return full


def bell(seat, center, width):
    """Normal density at an integer seat. Distant tails are dropped."""
    offset = (seat - center) / width
    if offset > 8.0 or offset < -8.0:
        return 0.0 * width
    body = chartmath.exp(-0.5 * offset * offset)
    return body / width


def hour_rows(packed, born, now, ahead, seats):
    """Densities for one hour, plus the fresh-title share held out of them.

    The jump is mixed in at once. A pace that makes the seat number worse
    is ignored for a jumper, because that lag is what the audience surge
    is supposed to close. The vote nudge moves only the diffused center.
    """
    width_h = WIDTH_FLOOR + WIDTH_SPAN * (1.0 - chartmath.exp(-ahead / WIDTH_HALFLIFE))
    stick = chartmath.exp(-ahead / STICK_HALFLIFE)
    fresh = min(FRESH_LIMIT, FRESH_FLOOR + FRESH_SLOPE * (ahead ** FRESH_POWER))
    rows = []
    for uid, feat, weight, aim, stale, vol in packed:
        alive = still_listed(uid, born, now, ahead)
        if alive <= 0.0:
            rows.append(None)
        else:
            pace = feat["pace"] * SLOPE_MEMORY * (1.0 - chartmath.exp(-ahead / SLOPE_MEMORY))
            if weight > 0.2 and pace > 0.0:
                pace = 0.0
            center = feat["seat"] + pace + feat["nudge"]
            if stale:
                center = center + AGED_SLIDE * ahead
            width = max(0.7, width_h * vol)
            jump_width = max(3.0, JUMP_WIDTH0 + 0.035 * ahead)
            stick_w = stick * (1.0 - 0.9 * weight)
            anchor = sticky_seat(feat["seat"])
            line = []
            seat = 1
            while seat <= seats:
                density = stick_w * bell(seat, anchor, STICK_WIDTH)
                wander = (1.0 - stick_w) * (1.0 - weight) * bell(seat, center, width)
                density = density + wander
                if weight > 0.02:
                    leap = (1.0 - stick_w) * weight * bell(seat, aim, jump_width)
                    density = density + leap
                line.append(alive * density)
                seat = seat + 1
            rows.append(line)
    return rows, fresh


def blend_hour(bucket, rows, fresh, seats):
    """Fold one hour into the running mixture, holding `fresh` out first."""
    seat = 0
    while seat < seats:
        total = 0.0
        for line in rows:
            if line is not None:
                total = total + line[seat]
        if total > 0.0:
            scale = (1.0 - fresh) / total
            index = 0
            while index < len(rows):
                line = rows[index]
                if line is not None:
                    bucket[index][seat] = bucket[index][seat] + line[seat] * scale
                index = index + 1
        seat = seat + 1


def label_chart(by_seat, packed):
    """Up to 100 rows a seat, naming every title on the newest chart.

    Mass under the log-score floor is dropped before the fill. Keeping it
    would mark the title as named, then rounding would delete the row and
    the title would dilute the share left for true newcomers.
    """
    listed = []
    place_index = 0
    for rows in by_seat:
        kept = []
        for uid, prob in rows:
            if prob >= FLOOR_P:
                kept.append((uid, prob))

        def heavier(pair):
            return (-pair[1], pair[0])

        kept.sort(key=heavier)
        mass = 0.0
        for _uid, prob in kept:
            mass = mass + prob
        if mass > 0.97:
            factor = 0.97 / mass
            resized = []
            for uid, prob in kept:
                resized.append((uid, prob * factor))
            kept = resized
            mass = 0.97
        seen = set()
        for uid, _prob in kept:
            seen.add(uid)
        for packed_row in packed:
            uid = packed_row[0]
            open_slot = len(kept) < ROW_CAP
            fresh_id = uid not in seen
            budget = (mass + FLOOR_P) <= 0.98
            if open_slot and fresh_id and budget:
                kept.append((uid, FLOOR_P))
                mass = mass + FLOOR_P
                seen.add(uid)
        seat_no = place_index + 1
        for uid, prob in kept:
            if prob >= FLOOR_P:
                listed.append((uid, seat_no, prob))
        place_index = place_index + 1
    return listed


def ordered_ten(triples, features, snapshots, seats):
    """Ten ids by conditional expected seat, with mass breaking ties."""
    mass = {}
    expect = {}
    for uid, seat, prob in triples:
        mass[uid] = mass.get(uid, 0.0) + prob
        expect[uid] = expect.get(uid, 0.0) + (seat * prob)

    def rank_key(uid):
        total = mass.get(uid, 0.0)
        if total <= 1e-6:
            missing = (50.0, 0.0, uid)
            return missing
        score = (expect[uid] / total, -total, uid)
        return score

    chosen = sorted(features, key=rank_key)
    chosen = list(chosen[:seats])
    if len(chosen) < seats:
        chart = snapshots[-1].get("up_and_coming") or []
        chart = sorted(chart, key=lambda entry: entry["rank"])
        for entry in chart:
            uid = entry["universe_id"]
            if uid not in chosen and len(chosen) < seats:
                chosen.append(uid)
    return chosen


def build_forecast(task, snapshots):
    """Average the hourly seat distributions across the scored window."""
    moments, tracks, born = build_tracks(snapshots)
    now = moments[-1]
    features = features_now(tracks, now)
    packed = pack_games(features)
    start = as_hours(task["window"]["from"])
    stop = as_hours(task["window"]["until"])
    slots = int(task["window"]["slots"])
    seats = int(task.get("places", SEAT_COUNT))
    width = (stop - start) / slots
    ahead = []
    cursor = 0
    while cursor < slots:
        mid = start + (cursor + 0.5) * width - now
        if mid < 0.3:
            mid = 0.3
        ahead.append(mid)
        cursor = cursor + 1
    bucket = []
    for _game in packed:
        bucket.append([0.0] * seats)
    for horizon in ahead:
        rows, fresh = hour_rows(packed, born, now, horizon, seats)
        blend_hour(bucket, rows, fresh, seats)
    count = float(len(ahead))
    by_seat = []
    seat = 0
    while seat < seats:
        by_seat.append([])
        seat = seat + 1
    game_index = 0
    for packed_row in packed:
        uid = packed_row[0]
        seat = 0
        while seat < seats:
            prob = bucket[game_index][seat] / count
            if prob > 0.0:
                by_seat[seat].append((uid, prob))
            seat = seat + 1
        game_index = game_index + 1
    triples = label_chart(by_seat, packed)
    top = ordered_ten(triples, features, snapshots, seats)
    filename = task.get("output", "forecast.json")
    return triples, top, filename


def quantize(triples):
    """Six decimals, at most 100 rows a seat, each seat summing to at most 1."""
    grouped = {}
    for uid, seat, prob in triples:
        rounded = round(float(prob), 6)
        grouped.setdefault(seat, []).append([uid, rounded])
    rows = []
    for seat in sorted(grouped):
        items = []
        for item in grouped[seat]:
            if item[1] >= FLOOR_P:
                items.append(item)

        def heavier(pair):
            return (-pair[1], pair[0])

        items.sort(key=heavier)
        if len(items) > ROW_CAP:
            items = items[:ROW_CAP]
        total = 0.0
        for item in items:
            total = total + item[1]
        if total > 1.0 and len(items) > 0:
            overflow = total - 1.0
            items[0][1] = round(items[0][1] - overflow, 6)
            if items[0][1] < FLOOR_P:
                items[0][1] = FLOOR_P
        for uid, prob in items:
            if prob >= FLOOR_P:
                record = {"universe_id": int(uid), "place": int(seat), "p": prob}
                rows.append(record)

    def document_order(row):
        return (row["place"], -row["p"], row["universe_id"])

    rows.sort(key=document_order)
    return rows


def run():
    task, snapshots = read_bundle()
    triples, top, filename = build_forecast(task, snapshots)
    names = []
    for uid in top:
        names.append(int(uid))
    document = {"probabilities": quantize(triples), "top10": names}
    chartos.makedirs(CHART_OUT, exist_ok=True)
    path = chartos.path.join(CHART_OUT, filename)
    with open(path, "w", encoding="utf-8") as handle:
        chartjson.dump(document, handle, separators=(",", ":"))
        newline = "\n"
        handle.write(newline)


if __name__ == "__main__":
    run()
