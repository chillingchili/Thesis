# Thesis APK Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. User instructed implementation after refreshing findings; proceed inline.

**Goal:** Deliver the updated thesis Android interface and a dependable on-device testing APK.
**Architecture:** Preserve the current inference and feedback policies. Add presentation formatting and session invalidation, move model work onto owned workers, and replace the debug overlay with camera-first result cards.
**Tech Stack:** Kotlin, XML/ViewBinding, Material Components, CameraX, MediaPipe, TFLite.
**Spec:** `docs/superpowers/specs/2026-10-12-thesis-android-apk-design.md`, as amended below and in `android/APP_READINESS.md`.

## Global constraints

- Native Android; existing original GRU weights and 0.60 classification threshold.
- Preserve GPU pose with CPU fallback, ServeWindow, MotionRate, default 350 ms hold, and the existing guarded general-mechanics feedback policy.
- General-mechanics cues require at least 32 real poses spanning 1500 ms; uncertain subtype remains explicit. Do not revert the latest policy to the frozen expert-study gate.
- Do not edit manuscript, responses, models, rule templates, or frozen evaluation artifacts.
- Offline; no account, cloud backend, or automatic video retention.
- Deliver a signed debug APK for thesis testing; separately verify release assembly.

## Review focus

1. Old callbacks after reset/background must not overwrite the current screen (session unit test and lifecycle phone test).
2. Low confidence with supported general cues must retain advice but mark subtype uncertainty (presentation test).
3. Missing or invalid metrics must not become a positive reference result (presentation test).
4. Camera permission refusal must leave recovery and research tools reachable (device test).
5. Model shutdown during active work must occur on the owning worker without recursive queue rejection (device lifecycle test and code review).

## Task 1: Refresh findings and presentation contract

Files: `android/APP_READINESS.md`, design amendment, `ServePresentation.kt`, `SessionEpoch.kt`, corresponding JVM tests.
Interfaces: `SessionEpoch.current`, `invalidate(): Long`, `accepts(token: Long): Boolean`; `ServePresentation.classification(label: String?, confidence: Float?): String`, `hip(value: Float?, passed: Boolean?): String`, `paddle(angle: Double?, state: FeedbackEngine.PaddleState?): String`.
- [ ] Record current thesis findings and explicit prior-finding corrections.
- [ ] Write tests for 0.5999/0.60 confidence, invalid scalars, and obsolete tokens; verify failure before implementation.
- [ ] Implement pure formatters and epoch guard; verify JVM suite.

## Task 2: Camera-first interface and dependable session

Files: `MainActivity.kt`, `PosePipeline.kt`, layouts, strings, colors, styles and drawable resources, manifest.
Interfaces: Existing binding IDs retained; new setup/tools/recovery/status/loading fields. Main activity consumes presentation helpers and epoch guard. Pose callbacks report missing-pose state and enqueue immutable landmark copies.
- [ ] Replace the overlay with separate preview and result cards, visible uncertainty, numeric measurements, optional landing state, and research-tools navigation.
- [ ] Add readable processing/ready/paused/error states, setup guidance and permission/settings recovery.
- [ ] Move model initialization/inference/closure to workers; guard UI callbacks, reset all cards together, close camera frames in finally, serialize pose start/analyze/stop.
- [ ] Preserve existing capture policy and exact feedback strings; use one completed result snapshot.
- [ ] Verify build/lint and unit suite; inspect the new screen on the phone.

## Task 3: Device verification and delivery

Files: `ThesisUiTest.kt`, `android/README.md`, `android/APP_READINESS.md`, local APK deliverable.
- [ ] Test actual activity launch, reset, background/foreground, tools/help, and recovery with instrumentation.
- [ ] Run existing JVM suite, debug/release assembly, lint, and selected phone UI tests.
- [ ] Inspect screenshots at normal and large text settings; correct clipping or contrast issues.
- [ ] Run one available live-path replay smoke test without replacing historical evidence; record limitations.
- [ ] Review diff with a fresh reviewer, fix substantive findings, and verify affected checks.
- [ ] Package signed testing APK, checksum, installation instructions and precise verification results.

## Execution record

- User's latest instruction authorizes updating findings then creating the app; no further design approval stop.
- Work stays in the user's existing checkout; untracked evaluation/training work belongs to the user and is preserved.
- Graphify guidance is inherited from `C:/Users/hibye/AGENTS.md`; no graph report/index exists in this repository at inspection time. Attempt update after implementation.
