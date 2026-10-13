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
)

/**
 * In-memory session log. Records stay in RAM only and vanish when the app
 * closes; "New attempt" and "Clear session" wipe them sooner. Video frames
 * are still discarded after each analysis cycle.
 */
object SessionStore {
    private val records = ArrayList<ServeRecord>()
    private var counter = 0

    @Synchronized fun nextNumber(): Int = ++counter
    @Synchronized fun add(record: ServeRecord) { records.add(record) }
    @Synchronized fun snapshot(): List<ServeRecord> = records.toList()
    @Synchronized fun size(): Int = records.size
    @Synchronized fun clear() { records.clear(); counter = 0 }

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
