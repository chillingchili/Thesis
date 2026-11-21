package com.thesis.pickleballserve

import android.content.res.ColorStateList
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TableRow
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.thesis.pickleballserve.databinding.ActivitySessionBinding
import java.util.Locale

/**
 * Serves review: every analyzed serve of this session with its clip (memory only).
 * Result shows the same reading as the live panel; Research shows this serve's
 * numbers against their references. Back returns to the live screen.
 */
class SessionActivity : AppCompatActivity() {

    private lateinit var b: ActivitySessionBinding
    private var records: List<ServeRecord> = emptyList()
    private var selected: Int = -1
    private var research = false
    private var slow = false
    private var playing = false
    private val handler = Handler(Looper.getMainLooper())

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun color(id: Int) = ContextCompat.getColor(this, id)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivitySessionBinding.inflate(layoutInflater)
        setContentView(b.root)
        enterImmersive(b.root)
        research = savedInstanceState?.getBoolean("research") ?: false
        b.backButton.setOnClickListener { finish() }
        b.tabGroup.check(if (research) R.id.tabResearch else R.id.tabResult)
        b.tabGroup.addOnButtonCheckedListener { _, id, checked ->
            if (!checked) return@addOnButtonCheckedListener
            research = id == R.id.tabResearch
            applyTab()
        }
        b.playButton.setOnClickListener { if (playing) pause() else play() }
        b.clearSessionButton.setOnClickListener {
            com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
                .setTitle(R.string.session_clear)
                .setMessage(R.string.session_clear_confirm)
                .setNegativeButton(android.R.string.cancel, null)
                .setPositiveButton(R.string.session_clear) { _, _ ->
                    SessionStore.clear(); pause(); records = emptyList(); selected = -1
                    buildStrip(); render()
                }.show()
        }
        b.rMoreCues.setOnClickListener { cuesExpanded = !cuesExpanded; records.firstOrNull { it.number == selected }?.let { bindCue(it) } }
        b.speedButton.setOnClickListener {
            slow = !slow
            b.speedButton.setText(if (slow) R.string.speed_slow else R.string.speed_full)
        }
        b.scrub.addOnChangeListener { _, value, fromUser ->
            if (fromUser) { pause(); seekTo(value.toLong()) }
        }
        records = SessionStore.snapshot().reversed()  // newest first
        selected = savedInstanceState?.getInt("selected", -1)?.takeIf { n -> records.any { it.number == n } }
            ?: records.firstOrNull()?.number ?: -1
        buildStrip()
        render()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean("research", research)
        outState.putInt("selected", selected)
        super.onSaveInstanceState(outState)
    }

    override fun onPause() { pause(); super.onPause() }

    // --- Layout: one fixed split for both tabs, so the clip never resizes. ---
    private fun applyTab() {
        val has = records.isNotEmpty()
        b.resultView.visibility = if (has && !research) View.VISIBLE else View.GONE
        b.researchView.visibility = if (has && research) View.VISIBLE else View.GONE
        b.emptyView.visibility = if (has) View.GONE else View.VISIBLE
        b.root.post { positionPeakTick() }
    }

    private fun buildStrip() {
        b.serveStrip.removeAllViews()
        for (r in records) {
            val chip = MaterialButton(this, null, 0).apply {
                setTextAppearance(R.style.TextAppearance_Serve_Button)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                isAllCaps = false
                cornerRadius = dp(10)
                insetTop = 0; insetBottom = 0
                minHeight = dp(44); minimumHeight = dp(44)
                setPadding(dp(12), 0, dp(12), 0)
                gravity = Gravity.CENTER
                val name = ServePresentation.serveTypeCard(r.label, r.confidence, r.probs?.toFloatArray()).value
                text = "${r.number}  $name" + (if (r.elapsed.isNotBlank()) "  ·  ${r.elapsed}" else "")
                contentDescription = "Serve ${r.number}, $name"
                tag = r.number
                setOnClickListener { selected = r.number; cuesExpanded = false; pause(); render() }
            }
            val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, dp(44))
            lp.marginEnd = dp(6)
            b.serveStrip.addView(chip, lp)
        }
        b.clearSessionButton.visibility = if (records.isNotEmpty()) View.VISIBLE else View.GONE
        styleStrip()
    }

    private fun styleStrip() {
        for (i in 0 until b.serveStrip.childCount) {
            val chip = b.serveStrip.getChildAt(i) as MaterialButton
            if (chip.tag == null) continue  // Clear session
            val on = chip.tag == selected
            val r = records.firstOrNull { it.number == chip.tag }
            val uncertain = r != null && !ServePresentation.serveTypeCard(r.label, r.confidence, null).reliable
            chip.backgroundTintList = ColorStateList.valueOf(color(R.color.surface))
            chip.strokeColor = ColorStateList.valueOf(color(if (on) R.color.select else R.color.border))
            chip.strokeWidth = dp(if (on) 2 else 1)
            chip.setTextColor(color(if (uncertain) R.color.warn else R.color.text_primary))
        }
    }

    private fun render() {
        applyTab()
        styleStrip()
        val r = records.firstOrNull { it.number == selected }
        if (r == null) {
            b.serveTitle.text = getString(R.string.no_serves_yet)
            b.clipView.clip = null
            showClipMessage(getString(R.string.review_empty_body))
            return
        }
        b.serveTitle.text = getString(R.string.serve_short, r.number)
        b.serveTitle.contentDescription = getString(R.string.serve_title, r.number, r.elapsed.ifBlank { r.time })
        bindClip(SessionStore.clip(r.number))
        renderResult(r)
        renderResearch(r)
    }

    // --- Clip playback (frame timestamps, optional 0.25x). ---
    private fun bindClip(clip: ServeClip?) {
        b.clipView.clip = clip
        if (clip == null) {
            showClipMessage(getString(R.string.no_clip)); return
        }
        b.clipEmpty.visibility = View.GONE
        b.clipControls.visibility = View.VISIBLE
        b.scrub.valueFrom = 0f
        b.scrub.valueTo = clip.durationMs.coerceAtLeast(1L).toFloat()
        b.scrub.value = 0f
        updateTime(0)
        b.root.post { positionPeakTick() }
    }

    private fun showClipMessage(text: String) {
        b.clipEmpty.text = text
        b.clipEmpty.visibility = View.VISIBLE
        b.clipControls.visibility = View.GONE
        b.peakTick.visibility = View.GONE
    }

    private fun positionPeakTick() {
        val clip = b.clipView.clip ?: return
        val peak = clip.peakMs
        if (peak == null || peak < clip.startMs || peak > clip.startMs + clip.durationMs || b.scrub.width == 0) {
            b.peakTick.visibility = View.GONE; return
        }
        val pad = b.scrub.trackSidePadding
        val frac = (peak - clip.startMs).toFloat() / clip.durationMs.coerceAtLeast(1L)
        b.peakTick.translationX = pad + (b.scrub.width - 2 * pad) * frac
        (b.peakTick.layoutParams as FrameLayout.LayoutParams).gravity = Gravity.START or Gravity.CENTER_VERTICAL
        b.peakTick.visibility = View.VISIBLE
        b.peakTick.contentDescription = "Peak motion"
    }

    private fun seekTo(offsetMs: Long) {
        val clip = b.clipView.clip ?: return
        val target = clip.startMs + offsetMs
        val idx = clip.frames.indexOfLast { it.tMs <= target }.coerceAtLeast(0)
        b.clipView.index = idx
        updateTime(clip.frames[idx].tMs - clip.startMs)
    }

    private fun updateTime(offsetMs: Long) {
        val clip = b.clipView.clip ?: return
        b.clipTime.text = String.format(Locale.getDefault(), getString(R.string.clip_time),
            offsetMs / 1000.0, clip.durationMs / 1000.0)
    }

    private val stepper = object : Runnable {
        override fun run() {
            val clip = b.clipView.clip ?: return pause()
            val i = b.clipView.index
            if (i >= clip.frames.size - 1) { pause(); return }
            val dt = clip.frames[i + 1].tMs - clip.frames[i].tMs
            b.clipView.index = i + 1
            val off = clip.frames[i + 1].tMs - clip.startMs
            b.scrub.value = off.toFloat().coerceIn(b.scrub.valueFrom, b.scrub.valueTo)
            updateTime(off)
            handler.postDelayed(this, (dt * if (slow) 4 else 1).coerceIn(10L, 1000L))
        }
    }

    private fun play() {
        val clip = b.clipView.clip ?: return
        if (b.clipView.index >= clip.frames.size - 1) seekTo(0)
        playing = true
        b.playButton.contentDescription = getString(R.string.pause)
        b.playButton.setIconResource(R.drawable.ic_pause)
        handler.post(stepper)
    }

    private fun pause() {
        playing = false
        handler.removeCallbacks(stepper)
        b.playButton.contentDescription = getString(R.string.play)
        b.playButton.setIconResource(R.drawable.ic_play)
    }

    /** Result tab cue: the first cue leads; further cues expand on request so the rows stay in view. */
    private var cuesExpanded = false
    private fun bindCue(r: ServeRecord) {
        val cues = r.feedback.lines().map { it.trim().removePrefix("• ") }.filter { it.isNotBlank() }
        if (cues.size <= 1 || cuesExpanded) b.rCue.setCueText(ServePresentation.coachingText(cues))
        else b.rCue.text = cues.first()
        b.rMoreCues.visibility = if (cues.size > 1) View.VISIBLE else View.GONE
        b.rMoreCues.text = if (cuesExpanded) getString(R.string.show_fewer_cues) else getString(R.string.show_all_cues, cues.size)
    }

    // --- Result tab: the live panel's reading for this serve. ---
    private fun renderResult(r: ServeRecord) {
        bindCue(r)
        b.rCueBadge.visibility = if (r.generalCue) View.VISIBLE else View.GONE
        val serve = ServePresentation.serveTypeCard(r.label, r.confidence, r.probs?.toFloatArray())
        val hip = ServePresentation.hipCard(r.hipValue, r.hipPassed)
        val paddle = ServePresentation.paddleCard(r.paddleAngle, r.paddleState, r.paddleLow, r.paddleHigh)
        b.rRows.removeAllViews()
        addRow(getString(R.string.serve_type_label), getString(R.string.gate_ref), serve.value,
            if (serve.reliable && r.confidence != null) "${ServePresentation.pct(r.confidence)}%" else null,
            when { serve.reliable -> null; serve.value == "Not measured" -> getString(R.string.badge_partial)
                   else -> getString(R.string.badge_low_confidence) },
            ServePresentation.Tone.WARN, warnValue = !serve.reliable)
        addRow(getString(R.string.hip_label),
            r.hipThreshold?.let { t -> r.hipValue?.let { getString(R.string.hip_ref, it, t) } ?: String.format(Locale.getDefault(), "≥ %.3f", t) } ?: "",
            hip.value, null, hip.chip?.let { if (it == "Within range") getString(R.string.badge_in_range) else it }, hip.tone)
        addRow(getString(R.string.paddle_label),
            if (r.paddleLow != null && r.paddleHigh != null) getString(R.string.paddle_ref, r.paddleLow, r.paddleHigh) else "",
            paddle.value, null,
            paddle.chip?.let { if (it == "Within range") getString(R.string.badge_in_range) else it } ?: getString(R.string.badge_partial),
            if (paddle.chip == null) ServePresentation.Tone.WARN else paddle.tone)
        r.landing?.let { zone ->
            b.rRows.addView(divider())
            b.rRows.addView(LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
                minimumHeight = dp(40)
                addView(text(getString(R.string.landing_label), R.style.TextAppearance_Serve_Label),
                    LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
                addView(text(zone, R.style.TextAppearance_Serve_Title).apply { setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f) })
            })
        }
    }

    private fun addRow(label: String, ref: String, value: String, suffix: String?, badge: String?,
                       tone: ServePresentation.Tone, warnValue: Boolean = false) {
        b.rRows.addView(divider())
        val row = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(4), 0, dp(6)) }
        row.addView(LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            addView(text(label, R.style.TextAppearance_Serve_Label),
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            addView(text(ref, R.style.TextAppearance_Serve_Label).apply {
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f); fontFeatureSettings = "tnum" })
        })
        row.addView(LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(2), 0, 0)
            addView(text(value, R.style.TextAppearance_Serve_Value).apply {
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 20f)
                if (warnValue) setTextColor(color(R.color.warn)) })
            suffix?.let { addView(text(it, R.style.TextAppearance_Serve_Label).apply { setPadding(dp(4), 0, 0, 0) }) }
            badge?.let { addView(badgeView(it, tone)) }
        })
        b.rRows.addView(row)
    }

    // --- Research tab: one table, three bars with one gate line, the cue source. ---
    private fun renderResearch(r: ServeRecord) {
        val t = b.researchTable
        t.removeAllViews()
        t.addView(tableRow(listOf(getString(R.string.col_measure), getString(R.string.col_value),
            getString(R.string.col_reference), getString(R.string.col_result)), header = true))
        val serve = ServePresentation.serveTypeCard(r.label, r.confidence, null)
        val conf = r.confidence?.takeIf { it.isFinite() }
        t.addView(tableRow(listOf(getString(R.string.serve_type_label),
            conf?.let { String.format(Locale.getDefault(), "%.2f", it) } ?: "—", "≥ 0.60", ""),
            badge = when {
                conf == null || serve.value == "Not measured" -> getString(R.string.res_not_measured) to ServePresentation.Tone.NEUTRAL
                conf >= FeedbackEngine.MIN_CONFIDENCE -> getString(R.string.res_passed) to ServePresentation.Tone.GOOD
                else -> getString(R.string.res_below_gate) to ServePresentation.Tone.WARN
            }))
        t.addView(tableRow(listOf(getString(R.string.hip_label),
            r.hipValue?.takeIf { it.isFinite() }?.let { String.format(Locale.getDefault(), "%.2f", it) } ?: "—",
            r.hipThreshold?.let { String.format(Locale.getDefault(), "≥ %.3f", it) } ?: "—", ""),
            badge = when (r.hipPassed) {
                null -> getString(R.string.res_not_measured) to ServePresentation.Tone.NEUTRAL
                true -> getString(R.string.res_pass) to ServePresentation.Tone.GOOD
                false -> getString(R.string.res_below) to ServePresentation.Tone.WARN
            }))
        t.addView(tableRow(listOf(getString(R.string.paddle_label),
            r.paddleAngle?.takeIf { it.isFinite() }?.let { String.format(Locale.getDefault(), "%.1f°", it) } ?: "—",
            if (r.paddleLow != null && r.paddleHigh != null)
                String.format(Locale.getDefault(), "%.1f–%.1f°", r.paddleLow, r.paddleHigh) else "—", ""),
            badge = when {
                r.paddleAngle == null || r.paddleState == null -> getString(R.string.res_not_measured) to ServePresentation.Tone.NEUTRAL
                r.paddleState == FeedbackEngine.PaddleState.OPTIMAL -> getString(R.string.res_within) to ServePresentation.Tone.GOOD
                else -> getString(R.string.res_outside) to ServePresentation.Tone.WARN
            }))

        b.probRows.removeAllViews()
        val probs = r.probs
        val names = listOf(getString(R.string.drive), getString(R.string.lob), getString(R.string.topspin))
        if (probs != null && probs.size == 3) {
            for (i in 0 until 3) b.probRows.addView(probRow(names[i], probs[i]))
        } else b.probRows.addView(text("—", R.style.TextAppearance_Serve_Small))

        val general = r.generalCue
        b.cueSource.text = getString(if (general) R.string.src_general else R.string.src_coach_rules)
        tintBadge(b.cueSource, if (general) ServePresentation.Tone.WARN else ServePresentation.Tone.GOOD)
        b.researchFoot.text = r.analysisMs?.let { getString(R.string.research_foot, it / 1000.0) }
            ?: getString(R.string.research_foot_plain)
    }

    private fun probRow(name: String, p: Float): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            minimumHeight = dp(26)
        }
        row.addView(text(name, R.style.TextAppearance_Serve_Label).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f); setTextColor(color(R.color.text_primary)) },
            LinearLayout.LayoutParams(dp(68), LinearLayout.LayoutParams.WRAP_CONTENT))
        val track = FrameLayout(this)
        track.addView(LinearProgressIndicator(this).apply {
            max = 1000; progress = (p.coerceIn(0f, 1f) * 1000).toInt()
            trackThickness = dp(8); trackCornerRadius = dp(4)
            setIndicatorColor(color(R.color.accent)); trackColor = color(R.color.accent_muted)
            trackStopIndicatorSize = 0; indicatorTrackGapSize = 0
            isIndeterminate = false
        }, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER_VERTICAL))
        val gate = View(this).apply { setBackgroundColor(color(R.color.warn)) }
        track.addView(gate, FrameLayout.LayoutParams(dp(2), dp(16), Gravity.CENTER_VERTICAL))
        // Gate line at 0.60 of the track, placed once the track has its width.
        track.addOnLayoutChangeListener { _, l, _, r, _, _, _, _, _ ->
            gate.translationX = (r - l) * FeedbackEngine.MIN_CONFIDENCE
        }
        row.addView(track, LinearLayout.LayoutParams(0, dp(20), 1f))
        row.addView(text(String.format(Locale.getDefault(), "%.2f", p), R.style.TextAppearance_Serve_Label).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f); gravity = Gravity.END; fontFeatureSettings = "tnum"
            setTextColor(color(R.color.text_primary)) },
            LinearLayout.LayoutParams(dp(48), LinearLayout.LayoutParams.WRAP_CONTENT))
        return row
    }

    private fun tableRow(cells: List<String>, header: Boolean = false,
                         badge: Pair<String, ServePresentation.Tone>? = null): TableRow {
        val row = TableRow(this)
        row.minimumHeight = dp(if (header) 28 else 40)
        row.gravity = Gravity.CENTER_VERTICAL
        cells.forEachIndexed { i, c ->
            if (i == 3 && badge != null) {
                val holder = LinearLayout(this).apply { gravity = Gravity.END or Gravity.CENTER_VERTICAL }
                holder.addView(badgeView(badge.first, badge.second).apply {
                    (layoutParams as? LinearLayout.LayoutParams)?.marginStart = 0 })
                row.addView(holder)
                return@forEachIndexed
            }
            row.addView(text(c, if (header) R.style.TextAppearance_Serve_Small else R.style.TextAppearance_Serve_Body).apply {
                setTextSize(TypedValue.COMPLEX_UNIT_SP, if (header) 13f else 15f)
                if (i == 1 && !header) setTextAppearance(R.style.TextAppearance_Serve_Title)
                if (i == 0 && !header) setTextColor(color(R.color.text_secondary))
                fontFeatureSettings = "tnum"
                setPadding(0, 0, dp(12), 0)
                if (i == 3) gravity = Gravity.END
            })
        }
        row.background = ContextCompat.getDrawable(this, R.drawable.row_divider)
        return row
    }

    private fun text(s: String, style: Int) = TextView(this).apply { setTextAppearance(style); text = s }

    private fun badgeView(label: String, tone: ServePresentation.Tone) = TextView(this).apply {
        text = label
        setTextAppearance(R.style.TextAppearance_Serve_Badge)
        background = ContextCompat.getDrawable(this@SessionActivity, R.drawable.badge)
        setPadding(dp(8), dp(3), dp(8), dp(3))
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginStart = dp(10) }
        tintBadge(this, tone)
    }

    private fun tintBadge(v: TextView, tone: ServePresentation.Tone) {
        val (bg, fg) = when (tone) {
            ServePresentation.Tone.GOOD -> R.color.ok_muted to R.color.ok
            ServePresentation.Tone.WARN -> R.color.warn_muted to R.color.warn
            ServePresentation.Tone.NEUTRAL -> R.color.accent_muted to R.color.text_secondary
        }
        v.backgroundTintList = ColorStateList.valueOf(color(bg))
        v.setTextColor(color(fg))
    }

    private fun divider() = View(this).apply {
        setBackgroundColor(color(R.color.border))
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(1))
    }
}
