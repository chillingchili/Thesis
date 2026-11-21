package com.thesis.pickleballserve

import android.app.Activity
import android.view.View
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat

/**
 * Full-screen landscape shell shared by every screen: system bars hide and come
 * back on a swipe; [root] keeps [gutterDp] plus any display-cutout inset.
 */
fun Activity.enterImmersive(root: View, gutterDp: Int = 8) {
    WindowCompat.setDecorFitsSystemWindows(window, false)
    WindowCompat.getInsetsController(window, root).apply {
        systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        hide(WindowInsetsCompat.Type.systemBars())
    }
    val gutter = (gutterDp * resources.displayMetrics.density).toInt()
    ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
        val cut = insets.getInsets(WindowInsetsCompat.Type.displayCutout())
        v.setPadding(gutter + cut.left, gutter + cut.top, gutter + cut.right, gutter + cut.bottom)
        insets
    }
}
