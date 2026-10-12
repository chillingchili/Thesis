package com.thesis.pickleballserve

import android.content.Intent
import android.os.SystemClock
import android.view.View
import android.widget.TextView
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test

/** Real activity tests: no synthetic inference outputs or UI-only demo fixtures. */
class ThesisUiTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()

    private fun awaitReady(scenario: ActivityScenario<MainActivity>) {
        val deadline = SystemClock.uptimeMillis() + 30_000
        var ready = false
        while (!ready && SystemClock.uptimeMillis() < deadline) {
            scenario.onActivity {
                ready = it.findViewById<View>(R.id.statusProgress).visibility != View.VISIBLE
            }
            if (!ready) SystemClock.sleep(100)
        }
        assertTrue("Initialization should finish or expose a recovery state", ready)
    }

    @Test fun screenRecoversAfterBackgroundAndReset() {
        val context = instrumentation.targetContext
        ActivityScenario.launch<MainActivity>(Intent(context, MainActivity::class.java)).use { scenario ->
            awaitReady(scenario)
            scenario.onActivity {
                assertEquals(android.content.res.Configuration.ORIENTATION_LANDSCAPE, it.resources.configuration.orientation)
                assertTrue(it.findViewById<View>(R.id.menuButton).isShown)
                assertTrue(it.findViewById<View>(R.id.serveInfoButton).isShown)
                assertTrue(it.findViewById<View>(R.id.hipInfoButton).isShown)
                assertTrue(it.findViewById<View>(R.id.paddleInfoButton).isShown)
                assertEquals("Waiting for a serve", it.findViewById<TextView>(R.id.classLabel).text.toString())
                assertEquals("Waiting for a serve", it.findViewById<TextView>(R.id.shiftLabel).text.toString())
                assertEquals(0, it.findViewById<android.widget.ProgressBar>(R.id.probDriveBar).progress)
                // Menu must open without crashing; dismiss it again.
                it.findViewById<View>(R.id.menuButton).performClick()
            }
            instrumentation.sendKeyDownUpSync(android.view.KeyEvent.KEYCODE_BACK)
            instrumentation.waitForIdleSync()
            repeat(2) {
                scenario.moveToState(Lifecycle.State.CREATED)
                scenario.moveToState(Lifecycle.State.RESUMED)
                awaitReady(scenario)
            }
            scenario.onActivity {
                assertFalse(it.isFinishing)
                assertNotEquals("Preparing analysis…", it.findViewById<TextView>(R.id.statusLine).text.toString())
            }
        }
        instrumentation.waitForIdleSync()
    }

    @Test fun missingResearchClipHasActionableRecovery() {
        val context = instrumentation.targetContext
        val intent = Intent(context, MainActivity::class.java).putExtra(EXTRA_REPLAY_CLIP, "missing-ui-test.mp4")
        ActivityScenario.launch<MainActivity>(intent).use { scenario ->
            awaitReady(scenario)
            scenario.onActivity {
                assertEquals("Replay unavailable", it.findViewById<TextView>(R.id.statusLine).text.toString())
                assertEquals(View.VISIBLE, it.findViewById<View>(R.id.recoveryButton).visibility)
                assertTrue(it.findViewById<View>(R.id.menuButton).isEnabled)
            }
        }
    }

    @Test fun interruptedCaptureDoesNotRemainAnalyzingOnResume() {
        val context = instrumentation.targetContext
        ActivityScenario.launch<MainActivity>(Intent(context, MainActivity::class.java)).use { scenario ->
            awaitReady(scenario)
            scenario.onActivity {
                // Exercise the real capture presentation without depending on a person serving.
                val render = MainActivity::class.java.getDeclaredMethod("clearCards", Boolean::class.javaPrimitiveType)
                render.isAccessible = true
                render.invoke(it, true)
                it.findViewById<View>(R.id.statusProgress).visibility = View.VISIBLE
            }
            scenario.moveToState(Lifecycle.State.CREATED)
            scenario.moveToState(Lifecycle.State.RESUMED)
            awaitReady(scenario)
            scenario.onActivity {
                assertNotEquals("Analyzing…", it.findViewById<TextView>(R.id.classLabel).text.toString())
                assertNotEquals("Analyzing…", it.findViewById<TextView>(R.id.shiftLabel).text.toString())
            }
        }
    }
}
