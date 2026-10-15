package com.thesis.pickleballserve

import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.card.MaterialCardView
import com.google.android.material.dialog.MaterialAlertDialogBuilder

/** Session serve log: review every analyzed serve without walking to the phone each time. */
class SessionActivity : AppCompatActivity() {

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun statusOf(record: ServeRecord): Pair<String, Int> {
        val known = record.serveType != "Unavailable" && record.serveType != "UNAVAILABLE"
        val conf = record.confidence?.takeIf { it.isFinite() && it in 0f..1f }
        return when {
            known && conf != null && conf >= FeedbackEngine.MIN_CONFIDENCE ->
                getString(R.string.status_completed) to R.color.accent
            known -> getString(R.string.status_low_confidence) to R.color.warning
            else -> getString(R.string.status_unavailable) to R.color.muted
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = getString(R.string.session_serves)
        val ink = ContextCompat.getColor(this, R.color.ink)
        val muted = ContextCompat.getColor(this, R.color.muted)
        val line = ContextCompat.getColor(this, R.color.line)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(ContextCompat.getColor(this@SessionActivity, R.color.page))
            setPadding(dp(16), dp(16), dp(16), dp(16))
        }
        val header = TextView(this).apply {
            textSize = 20f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setTextColor(ink)
        }
        val hint = TextView(this).apply {
            text = getString(R.string.session_hint)
            textSize = 14f
            setTextColor(muted)
            setPadding(0, dp(4), 0, dp(4))
        }
        val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        fun refresh() {
            val records = SessionStore.snapshot()
            header.text = if (records.isEmpty()) getString(R.string.session_empty)
            else getString(R.string.session_count, records.size)
            hint.visibility =
                if (records.isEmpty()) android.view.View.VISIBLE else android.view.View.GONE
            list.removeAllViews()
            for (record in records.reversed()) {
                val conf = record.confidence
                    ?.takeIf { it.isFinite() && it in 0f..1f }
                    ?.let { " · ${(it * 100).toInt()}%" } ?: ""
                val landing = record.landing?.let { " · Landing $it (second phone)" } ?: ""
                val (status, statusColor) = statusOf(record)
                val card = MaterialCardView(this).apply {
                    radius = dp(16).toFloat()
                    strokeColor = line
                    strokeWidth = dp(1)
                    cardElevation = 0f
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                    ).apply { topMargin = dp(12) }
                }
                val body = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(dp(16), dp(16), dp(16), dp(16))
                }
                body.addView(TextView(this).apply {
                    text = "Serve ${record.number} · ${record.time}"
                    textSize = 17f
                    setTypeface(typeface, android.graphics.Typeface.BOLD)
                    setTextColor(ink)
                })
                body.addView(TextView(this).apply {
                    text = status
                    textSize = 13f
                    setTypeface(typeface, android.graphics.Typeface.BOLD)
                    setTextColor(ContextCompat.getColor(this@SessionActivity, statusColor))
                    setPadding(0, dp(4), 0, 0)
                })
                body.addView(TextView(this).apply {
                    text = "${record.serveType}$conf$landing\n" +
                        "Hip: ${record.hip}\nPaddle: ${record.paddle}"
                    textSize = 14f
                    setTextColor(ink)
                    setPadding(0, dp(8), 0, 0)
                })
                body.addView(TextView(this).apply {
                    text = "${record.feedbackTitle}\n${record.feedback}"
                    textSize = 14f
                    setTextColor(ink)
                    setPadding(0, dp(8), 0, 0)
                })
                card.addView(body)
                list.addView(card)
            }
        }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val back = Button(this).apply {
            text = getString(R.string.back_to_live)
            setOnClickListener { finish() }
        }
        val guide = Button(this).apply {
            text = getString(R.string.setup)
            setOnClickListener {
                MaterialAlertDialogBuilder(this@SessionActivity)
                    .setTitle(R.string.setup_title)
                    .setMessage(R.string.setup_body)
                    .setPositiveButton(R.string.understood, null)
                    .show()
            }
        }
        val clear = Button(this).apply {
            text = getString(R.string.session_clear)
            setOnClickListener { SessionStore.clear(); refresh() }
        }
        buttons.addView(back)
        buttons.addView(guide)
        buttons.addView(clear)
        root.addView(header)
        root.addView(hint)
        root.addView(buttons)
        root.addView(ScrollView(this).apply {
            addView(list)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f).apply { gravity = Gravity.FILL }
        })
        setContentView(root)
        refresh()
    }
}
