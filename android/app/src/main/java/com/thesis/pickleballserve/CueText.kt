package com.thesis.pickleballserve

import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.LeadingMarginSpan
import android.widget.TextView

/**
 * Coaching cue for display: one cue reads as a plain sentence; several cues keep
 * their "• " bullets with a hanging indent, so wrapped lines align under the text.
 */
fun TextView.setCueText(cue: String) {
    val lines = cue.lines().filter { it.isNotBlank() }
    if (lines.size <= 1) { text = lines.firstOrNull()?.removePrefix("• ") ?: ""; return }
    val indent = paint.measureText("• ").toInt()
    val out = SpannableStringBuilder()
    lines.forEachIndexed { i, raw ->
        val line = if (raw.startsWith("• ")) raw else "• $raw"
        val start = out.length
        out.append(line)
        if (i < lines.size - 1) out.append('\n')
        out.setSpan(LeadingMarginSpan.Standard(0, indent), start, out.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    }
    text = out
}
