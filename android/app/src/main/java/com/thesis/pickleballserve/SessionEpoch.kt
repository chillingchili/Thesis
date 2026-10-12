package com.thesis.pickleballserve

import java.util.concurrent.atomic.AtomicLong

/** Invalidates asynchronous results without blocking the UI on model work. */
class SessionEpoch {
    private val value = AtomicLong()
    val current: Long get() = value.get()
    fun invalidate(): Long = value.incrementAndGet()
    fun accepts(token: Long): Boolean = token == value.get()
}
