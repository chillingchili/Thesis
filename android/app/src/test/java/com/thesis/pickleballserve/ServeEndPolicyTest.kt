package com.thesis.pickleballserve
import org.junit.Assert.*
import org.junit.Test
class ServeEndPolicyTest {
    @Test fun peakSettlingDoesNotCutImmediateFollowThrough() {
        assertFalse(ServeEndPolicy.quiet(.15f,.05f,1f,200))
        assertTrue(ServeEndPolicy.quiet(.15f,.05f,1f,400))
        assertFalse(ServeEndPolicy.quiet(.30f,.05f,1f,400))
        assertTrue(ServeEndPolicy.quiet(.01f,.05f,1f,200))
    }
}
