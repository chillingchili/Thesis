"""Python decision evaluator using the Android rule table verbatim."""
import math

ZONES = {"Deep-Left", "Deep-Right", "Short-Left", "Short-Right", "Fault-Long", "Fault-Wide", "Fault-Short"}


def evaluate(rules, serve_type, confidence, shift, paddle, placement_active=False, zone=None):
    notes = []
    if shift is None:
        notes.append("Weight shift unavailable.")
    if paddle is None:
        notes.append("Paddle orientation unavailable.")
    if placement_active:
        notes.append(f"Ball landing: {zone}." if zone in ZONES else "Ball landing unavailable.")

    def result(status, messages, ids=()):
        return dict(status=status, messages=messages, notifications=notes, rule_ids=list(ids),
                    version=rules["version"], validation_status=rules["validation_status"])

    subtype = (serve_type or "").lower()
    if subtype not in {"drive", "lob", "topspin"} or confidence is None or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return result("INSUFFICIENT_DATA", ["Serve classification unavailable. Record another serve for coaching feedback."])
    if confidence < 0.60:
        return result("LOW_CONFIDENCE", ["Serve classification confidence is below 60%. Record another serve for coaching feedback."])
    conditions = set()
    if shift is False:
        conditions.add("shift_insufficient")
    condition = {"OUTSIDE_BASELINE": "paddle_outside_baseline", "TOO_OPEN": "paddle_too_open",
                 "TOO_CLOSED": "paddle_too_closed"}.get(paddle)
    if condition:
        conditions.add(condition)
    matches = [r for r in rules["rules"] if subtype in r["serve_types"] and r["condition"] in conditions]
    if matches:
        return result("CORRECTIVE", [r["text"] for r in matches], [r["id"] for r in matches])
    if shift is True and paddle == "OPTIMAL" and (not placement_active or zone in ZONES and not zone.startswith("Fault-")):
        return result("POSITIVE", ["Good work: your measured weight shift and paddle angle are within the reference ranges. Keep practicing."], ["ASSESSED_MECHANICS_OK"])
    return result("PARTIAL", ["No supported corrective cue for the available measurements. Overall serve form has not been assessed."])
