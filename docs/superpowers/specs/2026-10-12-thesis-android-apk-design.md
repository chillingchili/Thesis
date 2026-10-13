# Thesis Android APK design

Date: 12 October 2026
Status: User instructed implementation after thesis refresh; amended below.

## Amendment after completed-thesis refresh

The user instructed: update findings first, then create the new UI and dependable
thesis APK. `android/APP_READINESS.md` is the current findings record and
`docs/superpowers/plans/2026-10-12-thesis-apk.md` is the execution plan.
They supersede conflicting pre-update findings below: GPU pose, time-normalized
motion and complete-window capture are now present. Preserve the new guarded
general-mechanics cues (32 real poses, 1500 ms) even below 0.60, while explicitly
marking subtype uncertainty. Do not revert the current app to unconditional
withholding. The completed thesis reports 18/30 mobile corrective outputs,
including 11 classification-independent cues, three missing captures and a
3.126-second-after-file-end case. These are documented limitations, not unfinished
thesis work. Frozen expert ratings do not validate the later mobile cues.

## Intended outcome

Deliver a polished, dependable Android APK for thesis use and on-device testing.
The user selected this scope rather than public Google Play distribution.
Use the supplied SRS and its wireframes as the starting point, incorporating the
newer thesis outputs and implementation decisions. Keep the existing native
Kotlin application and offline inference assets.

“Ready for thesis testing” means the agreed flows work, failure states are
recoverable, outputs accurately describe the measurements, and the APK has
build, automated-test, and physical-device evidence. It does not certify the
model for general coaching or establish unmeasured performance targets.

## Sources and precedence

1. The user's current scope and subsequent corrections.
2. `docs/thesis/Pascua-Leones_Thesis_Current.md`, particularly sections 4.4–4.5
   and Figures 8–10, plus `docs/PARTNER_HANDOFF.md` for completed evaluation.
3. The selected model and frozen feedback artifacts under
   `evaluation/coach_c/session2_gru/` and the deployed Android assets.
4. `SRS.md` for compatible requirements and the original visual arrangement.
5. Existing source and tests for observed behavior, not proof of requirements
   compliance.

Preserve the SRS, manuscript, model assets, response workbooks, and frozen
evaluation reports. Document requirement differences in the APK guide.

## Findings from the source review

| Area | Existing behavior | Design consequence |
|---|---|---|
| Platform | Kotlin, CameraX, MediaPipe, TFLite, OpenCV | Extend the existing app; do not restore the superseded React Native proposal. |
| Wireframes | Camera above result cards, with processing and unavailable states | Retain that hierarchy and adapt it to phone widths and large text. |
| Current home screen | Full-screen camera covered by developer labels and probability bars; five compact controls | Separate the visible preview from readable results and move research tools into a secondary entry. |
| Classifier | Original five-fold GRU default; single and hybrid comparison modes also present | Keep original GRU as the normal workflow and identify comparison modes clearly. |
| Hip output | Tick/cross under “WEIGHT SHIFT” | Show the measured hip-movement proxy, numeric value and reference state. It is not measured foot loading. |
| Paddle output | Reference state without the numeric angle on the main screen | Show available image-plane angle and within/outside-reference wording. |
| Confidence | Feedback engine applies 0.60; visual colors use other thresholds | Explicitly mark predictions below 0.60 as uncertain and withhold coaching through the existing gate. |
| Landing | Separate activity and input, inactive in core feedback | Show “Not included in this analysis”; keep optional landing review separate. Do not merge unrelated results. |
| Capture | Pose-motion start, low-motion end or timeout | Describe as motion-triggered live analysis. Player+paddle start and bounce+1-second end remain unmet SRS requirements. |
| Recording | Camera analysis requests 640×480; no demonstrated 1080p60 capture contract | Do not claim compliance with the old recording requirement. |
| Model initialization | GRU and some configuration load on the UI thread | Move expensive startup work off the UI thread with explicit loading/failure states. |
| Permission denial | Toast followed by activity finish | Provide an explanatory state with permission retry/settings recovery. |
| Worker lifetime | Main activity stops/closes workers and models without established worker ownership for all operations | Serialize model use and closure, bound queues, and invalidate obsolete callbacks. |
| Research replay | Separate model-comparison screen | Preserve as a research tool; do not present it as evaluated coaching parity. |
| Privacy | No network permission; backup enabled; research tools can explicitly export | Keep analysis offline, disable automatic app-data backup, and retain only user-requested exports or explicit research-tool outputs. |

The original SRS wireframes were visually inspected from their embedded PNGs.
They use a landscape camera region above four horizontal result cards. The
newer thesis adds hip movement and replaces unsupported form/direction labels.
Extracted reference images are in `outputs/app-readiness/` (generated files).

## Approach considered

**Recommended: modernize the current native app.** Reuse the evaluated models,
measurement logic, CameraX integration, and landing module. Introduce explicit
screen state and isolate the worker lifecycle where necessary. This keeps the
work aligned with the actual thesis implementation and existing test fixtures.

**Alternative: visual-only refresh.** Faster, but leaves permission recovery,
stale-result races, initialization blocking, and shutdown handling unresolved.
It does not meet the user's dependable-APK objective.

**Alternative: rebuild around the old SRS.** A React Native rewrite and new
bounce-synchronized capture would require substantial new integration and
validation. This changes the evaluated product beyond the selected thesis APK
scope and is not proposed for this release.

## Proposed experience

### Main analysis screen

Use a restrained, high-contrast Material interface. Place the app title and a
short textual state above the camera preview. Keep the camera visibly distinct
from the results. Offer setup guidance and a secondary Research tools menu.
The normal player workflow uses the original GRU without model-selection jargon.

Place results below the preview in this order:

1. **Serve type:** Drive, Lob or Topspin; numeric confidence; explicit uncertain
   status below 0.60. Never label the whole serve Good Form or Bad Form.
2. **Hip movement:** normalized image-plane displacement, reference comparison,
   and concise explanation that this is a weight-shift proxy.
3. **Paddle orientation:** numeric image-plane angle when available, with
   within-reference or outside-reference state. Do not infer physical opening
   or closing from unvalidated measurements.
4. **Coaching:** the existing deterministic rule output, limited positive cue,
   uncertainty notice, partial notice, or unavailable state.
5. **Ball landing:** a compact optional-output state. Explain that landing is
   not part of this analysis; offer the separate review tool. Only show a court
   marker for a real result in that tool, never an invented main-screen result.

Use stacked cards on narrow displays. On wider displays, arrange the compact
measurement cards in columns and give coaching sufficient width. Use scrolling
for long text and preserve system-inset clearance, including Android 15 edge
handling. Avoid fixed-height text containers and tiny five-button rows.

### States and actions

Use explicit initializing, permission-required, ready, analyzing, complete,
partial, low-confidence, and recoverable-error states. Display loading
placeholders only while work is active; never leave an endless calculation state
after a failure. Missing measurements are per-card states.

Hold a completed result until the next capture starts or the user resets it.
At the next capture, clear every old result and probability together. A reset,
pause, new capture, or shutdown invalidates callbacks from the previous run.
Keep recovery actions reachable when a model or camera is unavailable.

Use the actual motion-triggered behavior for the live workflow. Keep a manual
reset/retry action for recovery. Setup help explains right-handed support,
full-body and paddle visibility, side-view placement, and the study's fixed
camera setup. It must not advertise compliance with unimplemented automatic
capture or recording specifications.

### Research tools

Move single/hybrid comparison selection, complete-serve model comparison,
optional landing analysis, probability diagnostics, and latency tests behind a
clearly labeled secondary entry. Retain their function and existing evidence
contracts. Do not silently switch the normal flow to the research model.

Recorded model comparison is not a new unified recorded-coaching flow. Building
such a flow would require an explicit preprocessing/window contract and parity
evidence; it is outside this proposed iteration.

## Architecture and lifecycle

Keep XML/ViewBinding and Material Components; no new UI framework is required.
Separate immutable display state and presentation formatting from activity
rendering. Use a small session/controller boundary for generation IDs, loading,
reset, and lifecycle transitions. Keep the current numerical preprocessing,
selected models, threshold files, and rule text unchanged.

An analysis result must snapshot subtype, confidence, hip metric/state, paddle
metric/state, and feedback from one completed session before buffers clear.
The UI consumes that snapshot rather than independently reading mutable buffers.
Workers attach the active generation to work; stale jobs cannot update a new
session's measurements or result cards.

Load and invoke models on their owning workers. Queue their closure after
in-flight work has finished. Shutdown rejects new work without recursive
resubmission. Bound frame retention and queued detection work. Stop analysis
when the app leaves the foreground and resume to a coherent state. Ensure every
camera frame closes even when conversion or downstream processing fails.

Handle absent pose frames and loss of subject explicitly so the last “Pose OK”
message and active capture do not remain indefinitely. Preserve valid independent
measurements on partial failures; do not convert missing values into a pass.

## Packaging and documentation

Deliver an installable, locally signed test APK using the project's existing
debug signing path, labeled as a thesis testing build. Also verify the release
build can assemble; release-key creation and public distribution are not part of
this scope. Report the actual APK path, build variant, and checksum.

Make internal activities non-exported unless a documented testing dependency
requires otherwise. Check test entry points before changing them. Disable
automatic backup and avoid automatic video/history retention in normal use.
Do not delete user-owned imported videos or explicit research exports.

Update `android/README.md` with installation, normal use, research tools,
permissions, offline operation, known requirement differences, and actual
verification results. Preserve model/rule provenance without claiming every cue
has been independently certified from aggregate Coach C responses.

## Acceptance criteria

- Main screen reflects the original camera/results hierarchy and all applicable
  amended outputs. Processing, completed, uncertain and unavailable states are
  readable and distinct in words as well as color.
- Original GRU remains the normal classifier, with the unchanged 0.60 gate.
- Numeric hip and paddle values are shown only when valid; available outputs
  survive an unrelated stream failure.
- No unsupported whole-form verdict, physical face direction, fabricated
  landing point, or live-performance claim appears in the normal interface.
- Permission denial and camera/model failures expose a usable recovery path.
- Reset, background/foreground, repeated serves, and exit during analysis do
  not crash or display obsolete results.
- Text remains usable with large font settings and small-screen layouts;
  interactive targets have accessible labels and adequate size.
- Existing numerical and feedback regression tests pass, plus focused tests
  for state transitions, invalidation, partial results and confidence boundary.
- Debug unit tests, APK assembly and lint pass; release assembly is checked.
- Connected-device checks exercise launch, permission recovery, live preview,
  repeated lifecycle transitions, research navigation, and available replay
  fixtures. Record which checks are automated and which need a person serving.
- APK installation and launch are verified on the connected Redmi Note 12 when
  device access remains available.

## Release evidence and limits

The existing benchmark reports approximately 68.6 seconds median for recorded
file processing. That interval differs from live capture-end-to-display latency.
The live two-second requirement remains unverified and cannot be established by
UI polishing or replay alone. Motion-triggered live windows also differ from
the frozen recorded-clip evaluation; do not claim identical accuracy or outputs.

The repository handoff reports Session 2 received, with 27 Valid responses out
of 30 and documented response-category interpretation limits. Historical model
cross-validation and those ratings do not establish general-user reliability.

No model retraining, threshold tuning, frozen-report regeneration, coach-response
editing, public publishing, or thesis conclusion changes belong to this APK work.

## Review decision

### Baseline verification before implementation

On 12 October 2026, the existing project passed
`:app:testDebugUnitTest :app:assembleDebug :app:lintDebug` using JBR 21 and
Gradle 8.9. The unit-test reports contain 35 tests, zero failures, zero errors,
and zero skipped tests. Lint reports 123 warnings and no errors: predominantly
runtime-composed text (80) and hardcoded text (19). A passing lint task is not a
warning-free baseline. The connected-device inventory includes Redmi test
device `1fb637d`; no new APK was installed or device test run during this design
review. No product code has changed.

Approve or amend the native Android design above, particularly the distinction
between a dependable thesis APK and the still-unmet automatic capture/latency
requirements. After design approval, prepare the concrete implementation plan
and verification steps under the development workflow.
