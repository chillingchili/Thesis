# Thesis APK: refreshed findings and delivery record

## Findings refreshed from the completed thesis

Reviewed 12 October 2026 against commit `c9f4db0e6`, the current manuscript,
thesis README post-review update, and `evaluation/coach_c/mobile_gpu_rate_20261012/REPORT.md`.

The earlier app design described the pre-update implementation. These corrections
supersede those findings:

- The thesis now reports the completed mobile follow-up, not just the old
  68.6-second recorded-file benchmark. The later live-path development comparison
  produced 2/30 corrective/positive outputs on CPU, 5/30 with GPU alone, and 18/30
  after GPU acquisition plus automatic trigger/general-cue changes.
- The mobile implementation uses GPU pose with CPU fallback, time-normalized
  motion, one-second real-pose pre-roll, a complete serve window, and end-only
  classifier inference. The selected original GRU weights remain unchanged.
- The default end hold remains 350 ms. The longer-hold pilot did not resolve
  missed captures or the late case and is not the selected default.
- The 0.60 threshold still determines reliable subtype classification. The mobile
  development policy permits existing rules shared by every subtype when at
  least 32 real poses span 1500 ms. Eleven of the 18 mobile cues used this guarded
  general-mechanics path. The UI must preserve these cues and mark uncertainty.
- The frozen Coach C study and later mobile outputs are separate. The frozen
  study received 27 Valid, one Not valid and two no-coaching responses. Later
  mobile cues were not independently rated; old ratings must not be transferred.
- Three clips produced no completed mobile output, and one result arrived 3.126 s
  after file end. Completion of the thesis does not mean every production target
  was achieved. These are reported limitations, not missing thesis chapters.
- Landing remains optional and separate. Hip movement is an image-plane proxy;
  paddle output is reference-band membership. No whole-serve Good/Bad Form or
  validated physical paddle-face opening/closing verdict is introduced.

## Implementation scope

User correction: the interface must be landscape, matching the supplied
wireframes. Camera spans the top; Serve Type, Paddle Angle, Ball Landing and
Feedback form four columns below. Hip movement is included in the Serve Type
column. Each result column scrolls independently for long content and large text.

Polish and harden the native Android APK for thesis use and on-device testing:
camera-first interface, accessible result cards, exact current rule messages,
explicit uncertainty and partial data, permission recovery, worker lifecycle,
and secondary research tools. Keep research evidence and model assets unchanged.

The original SRS's React Native stack and unconditional coaching withholding are
superseded by the current implementation. Player+paddle automatic start,
bounce-based stop, 1080p60 capture, synchronized auxiliary input, and universally
reliable two-second feedback are not claimed by this release.

## Verification

Verified 12 October 2026 on JDK 17 (Gradle 8.9, `--offline`):

- `:app:testDebugUnitTest` — 47/47 pass (FeedbackEngine 13, ServePresentation 4
  incl. SessionEpoch, paddle/hip/classification guards, plus hybrid, motion,
  window, landing suites). No failures or errors.
- `:app:lintDebug` — success, zero Error/Fatal issues (94 pre-existing
  warnings: i18n/hardcoded-text, dependency, unused-resource classes).
- `:app:assembleDebug` + `:app:assembleRelease` — both succeed.
  `app-debug.apk` badging confirms package `com.thesis.pickleballserve`,
  `versionCode=2 versionName=1.1-thesis`, `minSdk 26 / targetSdk 35`, and a
  camera-only permission set (`CAMERA` plus the app's own dynamic-receiver
  permission; transitive `INTERNET`/storage/phone-state permissions are
  stripped in the manifest, enforcing the offline thesis claim).
- Landscape is asserted by the `ThesisUiTest` activity checks
  (`ORIENTATION_LANDSCAPE`, header controls and all four cards visible);
  connected phone tests were not re-run here — run them with the phone
  unlocked per `android/README.md` before the defense demo.

Delivered in this pass: committed Gradle wrapper (`gradlew`/`gradlew.bat`,
Gradle 8.9) so the APK rebuilds outside Android Studio; landscape lock on all
four activities; back/front camera input selector; (?) reference definitions
on every result card (serve subtypes, hip proxy, paddle reference band,
landing scope); analyzing/insufficient-data states on all four cards;
Back-to-live navigation on every research tool; camera-only permission
hardening; sideload install notes in `android/README.md`.

Revision, later 12 October 2026 (installed and screenshot-verified on the
Redmi Note 12, `versionCode=2 versionName=1.1-thesis`): the header is now a
slim title bar with a single Menu (setup, new attempt, back/front camera
radio, probabilities, compare, landing, latency, model, about); the status
strip moved above the camera feed with no overlays on the preview; Hip
Movement is a fifth column so all five outputs are visible with no scrolling
(scroll kept only as a large-text fallback). Unit tests, lint (zero errors)
and debug/release assembly re-verified after the change.

Revision, evening 12 October 2026 (screenshot-verified on-device again):
Ball Landing removed from the main screen — four columns remain (Serve
Type, Hip Movement, Paddle Angle, Feedback). The camera is now full-bleed
wide (edge to edge, fill instead of letterboxed fit, no rounded box); only
a compact status pill and the Menu float over it. The landing review screen
was updated to match: wide preview, top bar (back, title, zone-legend ?,
load video), live court overlay and status. Tests, lint and assembly green.

Revision, night 12 October 2026 (screenshot-verified on-device): tap-the-
preview / Menu fullscreen camera for tripod framing (Back exits); Menu →
Session serves keeps an in-memory log of every analyzed serve with its
measurements and coaching cue plus Clear; results are spoken aloud through
on-device TTS (Menu toggle, same strings as on screen) so the player never
walks to the phone mid-session. 48/48 unit tests pass, lint zero errors,
debug/release assemble.
