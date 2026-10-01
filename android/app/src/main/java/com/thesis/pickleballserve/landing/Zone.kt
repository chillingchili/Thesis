package com.thesis.pickleballserve.landing

object Zone {
    val KPT_COURT_FT: DoubleArray = doubleArrayOf(
        0.0, 0.0, 20.0, 0.0, 20.0, 44.0, 0.0, 44.0,
        0.0, 15.0, 20.0, 15.0, 0.0, 29.0, 20.0, 29.0,
        10.0, 0.0, 10.0, 22.0, 10.0, 29.0, 10.0, 44.0,
        0.0, 22.0, 20.0, 22.0,
    )

    val SKELETON_EDGES = listOf(
        0 to 8, 8 to 1, 0 to 4, 4 to 12, 12 to 6, 6 to 3,
        1 to 5, 5 to 13, 13 to 7, 7 to 2, 4 to 5,
        6 to 10, 10 to 7, 12 to 9, 9 to 13, 3 to 11, 11 to 2,
    )

    const val BEV_SCALE = 14f
    const val BEV_MARGIN = 30f
    const val BEV_W = 20 * BEV_SCALE + 2 * BEV_MARGIN
    const val BEV_H = 44 * BEV_SCALE + 2 * BEV_MARGIN

    fun classify(xFt: Double, yFt: Double): String {
        if (xFt < 0 || xFt > 20) return "Fault-Wide"
        if (yFt < 0 || yFt > 44) return "Fault-Long"
        if (yFt >= 15 && yFt <= 29) return "Fault-Short"
        val side = if (xFt < 10) "Left" else "Right"
        val depth = if (yFt < 15) {
            if (yFt < 7.5) "Deep" else "Short"
        } else {
            if (yFt > 36.5) "Deep" else "Short"
        }
        return "$depth-$side"
    }
}
