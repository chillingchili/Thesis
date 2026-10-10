# Coach C Session 2 Google Form

Copy the complete `coach_c_session2.gs` file into a Google Apps Script project.
It includes the exact approved wording-v2 output for C001–C030, including notices.
No CSV upload or manual entry of feedback is necessary.

1. Enable **Services → + → YouTube Data API v3** in Apps Script.
2. Check `C2_PLAYLIST_ID`. The supplied `PLdBi5bcaLX50` value is retained; replace it if needed with the complete value after `list=` in the playlist URL.
3. Run `verifyCoachCSession2Playlist()` and authorize the requested Google access.
4. Run `createCoachCSession2Form()` three times. It adds ten clips per execution and opens responses after all 30 are complete. Interrupted batches can be rerun.
5. Find edit/respondent links in the execution log or run `showCoachCSession2Links()`. Preview the video playback and form before sharing the respondent link.

If the last execution stops after building all items but before opening responses,
run `publishCoachCSession2Form()` to finish publication. Google Workspace access
restrictions may require adjusting responder access in the Forms editor.

The generated form presents each anonymous clip, its model feedback, and one
required question with three choices:

- Valid
- Not valid
- No coaching feedback to assess

The third choice is for outputs that only withhold coaching, report unavailable
measurements, or say no corrective cue is supported. Missing paddle information
alongside an actual correction does not automatically make the feedback unassessable.
No option is preselected. There is no Good/Bad Form or deviation-comment question.

All state uses separate Session 2 names. Session 1 is not edited.
To update an existing form, replace the entire script and run
`updateCoachCSession2Display()`. It removes serve types from video titles and clip
headings, then displays `Serve classification: Drive | Feedback: ...` in each
model-output block. The classification comes from the frozen hybrid prediction,
not the recorded category. Below-threshold predictions include `(low confidence)`.
Existing feedback wording and question names stay unchanged. Later batches use
the same format. The old `renameClipsByServeType()` name delegates to this update.
Do not keep duplicate copies of these functions in the Apps Script project.
Do not modify embedded feedback after collection starts.

## Analysis

This instrument adds an explicit no-coaching category at the researcher's request.
Report Valid, Not valid and No coaching feedback to assess counts out of all 30.
Report Valid/30 as the full-set percentage. If also reporting validity among
assessed outputs, state that denominator separately. Do not score the third option
as Valid or silently remove it. The underlying output coverage stays 16 corrective,
one positive, 12 low-confidence and one partial output; coach ratings are independent.
This revised instrument does not separately judge whether abstention itself was appropriate.
The researcher requested classification to be displayed alongside feedback; this
changes the presentation from the original feedback-only packet. Keep track of
which presentation version Coach C receives. A predicted serve label alone does
not count as substantive coaching for the third response option.

Source: `outputs/coach_c_evaluation_v2/private_session2_wording_v2_20261010T080226Z/session2_packet_HOLD/feedback_strings.csv`.
CSV SHA-256: `5a3cbc82482eee1876a1f50c76451f4886b27d372f512126e3e2b8937ffb5d4f`.

Local verification: `node scripts/google_apps_script/test_coach_c_session2.cjs`.
This tests mocked Apps Script services; the script has not been executed in the user's Google account.

Google references: [YouTube advanced service](https://developers.google.com/apps-script/advanced/youtube),
[Forms API](https://developers.google.com/apps-script/reference/forms/form),
[Video items](https://developers.google.com/apps-script/reference/forms/video-item).
