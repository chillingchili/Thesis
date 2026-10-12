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
}
