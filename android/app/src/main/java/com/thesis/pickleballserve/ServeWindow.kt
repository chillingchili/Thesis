package com.thesis.pickleballserve

/** Real timestamped pose samples; pre-roll never includes a previous active serve. */
class ServeWindow(private val preRollMs: Long = 1000L, private val capacity: Int = 512) {
    data class Sample(val timeMs: Long, val features: FloatArray)
    data class Input(val tensor: Array<Array<FloatArray>>, val validFrames: Int)
    private val samples = ArrayDeque<Sample>()
    var active = false
        private set
    val size: Int get() = samples.size
    val durationMs: Long get() = if (samples.size < 2) 0L else samples.last().timeMs - samples.first().timeMs

    fun add(timeMs: Long, features: FloatArray) {
        require(features.size == 10)
        if (samples.isNotEmpty() && timeMs <= samples.last().timeMs) return
        samples.addLast(Sample(timeMs, features.copyOf()))
        if (!active) while (samples.isNotEmpty() && timeMs - samples.first().timeMs > preRollMs) samples.removeFirst()
        while (samples.size > capacity) samples.removeFirst()
    }

    fun begin() { active = true }
    fun clear() { samples.clear(); active = false }

    /** Keep all real samples when short; select unique samples over elapsed time when long. */
    fun input(length: Int = 128): Input {
        require(length >= 2)
        val source = samples.toList()
        val count = minOf(source.size, length)
        val output = Array(1) { Array(length) { FloatArray(10) } }
        var previous = -1
        for (i in 0 until count) {
            val index = if (source.size <= length) i else {
                val target = source.first().timeMs + durationMs * i / (length - 1)
                val first = previous + 1
                val last = source.size - (length - i)
                (first..last).minBy { kotlin.math.abs(source[it].timeMs - target) }
            }
            output[0][i] = source[index].features.copyOf()
            previous = index
        }
        return Input(output, count)
    }
}
