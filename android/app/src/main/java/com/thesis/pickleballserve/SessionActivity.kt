package com.thesis.pickleballserve

import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/** Session serve log: review every analyzed serve without walking to the phone each time. */
class SessionActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = getString(R.string.session_serves)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
        }
        val header = TextView(this).apply { textSize = 20f }
        val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        fun refresh() {
            val records = SessionStore.snapshot()
            header.text = if (records.isEmpty()) getString(R.string.session_empty)
            else getString(R.string.session_count, records.size)
            list.removeAllViews()
            if (records.isEmpty()) {
                list.addView(TextView(this).apply {
                    text = getString(R.string.session_hint)
                    textSize = 15f
                })
            }
            for (record in records.reversed()) {
                list.addView(TextView(this).apply {
                    val conf = record.confidence?.let { " · ${(it * 100).toInt()}%" } ?: ""
                    text = ("Serve ${record.number} · ${record.time}\n" +
                        "${record.serveType}$conf\n" +
                        "Hip: ${record.hip}\nPaddle: ${record.paddle}\n" +
                        "${record.feedbackTitle}: ${record.feedback}")
                    textSize = 15f
                    setPadding(0, 12, 0, 12)
                })
            }
        }
        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val back = Button(this).apply {
            text = getString(R.string.back_to_live)
            setOnClickListener { finish() }
        }
        val clear = Button(this).apply {
            text = getString(R.string.session_clear)
            setOnClickListener { SessionStore.clear(); refresh() }
        }
        buttons.addView(back)
        buttons.addView(clear)
        root.addView(header)
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
