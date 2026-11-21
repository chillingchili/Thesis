package com.thesis.pickleballserve

/** One analyzed serve. Display-only; inference and feedback policy are untouched. */
data class ServeRecord(
    val number: Int,
    val time: String,
    val serveType: String,
    val confidence: Float?,
    val hip: String,
    val paddle: String,
    val feedbackTitle: String,
    val feedback: String,
    /** Ball-landing zone scanned from the second phone's QR code, null when unavailable. */
    val landing: String? = null,
    // --- Raw measurements for the review screen's Result and Research tabs. ---
    /** Session clock at the serve, e.g. "0:24". */
    val elapsed: String = "",
    /** Raw GRU label ("drive" / "lob" / "topspin"), null when not measured. */
    val label: String? = null,
    /** Class probabilities in drive, lob, topspin order. */
    val probs: List<Float>? = null,
    val hipValue: Float? = null,
    val hipPassed: Boolean? = null,
    val hipThreshold: Float? = null,
    val paddleAngle: Double? = null,
    val paddleState: FeedbackEngine.PaddleState? = null,
    val paddleLow: Double? = null,
    val paddleHigh: Double? = null,
    /** Cue came from the mobile general-mechanics policy (subtype withheld below the gate). */
    val generalCue: Boolean = false,
    /** Serve end to result, in ms. */
    val analysisMs: Long? = null,
)

/**
 * In-memory session log. Records and clips stay in RAM only and vanish when
 * the app closes; "Clear session" wipes them sooner. Nothing is written to storage.
 */
object SessionStore {
    private val records = ArrayList<ServeRecord>()
    private val clips = LinkedHashMap<Int, ServeClip>()
    private var counter = 0
    private var startMs = 0L

    /** Clips kept for the most recent serves only (about 2 MB each). */
    const val MAX_CLIPS = 20

    @Synchronized fun nextNumber(): Int = ++counter
    @Synchronized fun add(record: ServeRecord) { records.add(record) }
    @Synchronized fun snapshot(): List<ServeRecord> = records.toList()
    @Synchronized fun size(): Int = records.size
    @Synchronized fun clear() { records.clear(); clips.clear(); counter = 0; startMs = 0L }

    /** Session clock starts at the first live session of this log. */
    @Synchronized fun markStart(nowMs: Long) { if (startMs == 0L) startMs = nowMs }
    @Synchronized fun elapsedLabel(nowMs: Long): String {
        if (startMs == 0L) startMs = nowMs  // first serve after Clear session restarts the clock
        val s = if (startMs == 0L) 0L else ((nowMs - startMs) / 1000L).coerceAtLeast(0L)
        return "${s / 60}:${(s % 60).toString().padStart(2, '0')}"
    }

    @Synchronized fun putClip(number: Int, clip: ServeClip) {
        clips[number] = clip
        while (clips.size > MAX_CLIPS) clips.remove(clips.keys.first())
    }
    @Synchronized fun clip(number: Int): ServeClip? = clips[number]

    /**
     * Amend a logged serve (dual-phone landing attached after analysis).
     * Returns false when the serve number is unknown.
     */
    @Synchronized fun update(number: Int, transform: (ServeRecord) -> ServeRecord): Boolean {
        val index = records.indexOfFirst { it.number == number }
        if (index < 0) return false
        records[index] = transform(records[index])
        return true
    }
}
