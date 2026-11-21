package com.thesis.pickleballserve

import android.app.Application
import androidx.appcompat.app.AppCompatDelegate

/** Applies the saved Theme choice (System default / Light / Dark) before any screen opens. */
class ServeApp : Application() {
    override fun onCreate() {
        super.onCreate()
        AppCompatDelegate.setDefaultNightMode(
            getSharedPreferences(PREFS, MODE_PRIVATE).getInt(KEY_NIGHT_MODE, AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM)
        )
    }

    companion object {
        const val PREFS = "display"
        const val KEY_NIGHT_MODE = "night_mode"
    }
}
