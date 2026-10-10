# Coach C evaluation — current handoff

Session 1 is complete under the actual form-and-comment protocol: 27 Bad Form,
three Good Form (C008, C019, C022). Independent subtype labels were intentionally
omitted. Session 2 ratings have not been received in this workspace.

Use the [current Apps Script](../../scripts/google_apps_script/README.md) for
Session 2. It shows clip-only titles, frozen predicted subtype plus approved
feedback, and Valid / Not valid / No coaching feedback to assess choices.
The classification display was requested after the original feedback-only ZIP.

```text
coach_c/
├── README.md              Current interpretation and workflow
├── provenance.json        Original paths, SHA-256 hashes and byte counts
├── source/                Selection, researcher key, technical checks, latest status
├── session1/              Original response workbook and normalized annotations
├── baseline/              Original frozen inference, manifest and 30 reports
├── session2/              Approved revised wording, 30 reports and scope amendment
└── videos/                One exact copy of each selected C001–C030 recording
```

Historical files are intentionally preserved. In particular, the first Session 1
audit and original freeze files contain now-superseded HOLD instructions and the
old planned subtype-label requirement. Read `session2/EVALUATION_SCOPE_AMENDMENT.md`
for the researcher's correction. Source status reports readiness, not proof that
a form or packet was sent to Coach C. No Google deployment is verified locally.

All 30 produce predictions. Outputs comprise 16 corrective, one positive,
12 low-confidence and one partial result. Keep all 30 in Session 2 accounting.
Report Valid/30 and all response categories; any assessed-only rate is secondary.
No whole-form accuracy, independent Coach C classification accuracy or feedback
appropriateness result can be inferred from the Session 1 Bad/Good Form counts.

`baseline/` and `session2/reports/` retain their original source-layout paths and
freeze references. Use `provenance.json` to locate the corresponding handoff copies.
Videos are included once; repeated 149 MB ZIPs and generated previews remain local.
The local archive under `outputs/coach_c_evaluation_v2/` remains unchanged.

For the wider project and remaining thesis work, read
[Partner handoff](../../docs/PARTNER_HANDOFF.md).
