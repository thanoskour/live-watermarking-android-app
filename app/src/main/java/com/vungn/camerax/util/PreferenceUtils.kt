package com.vungn.camerax.util

import android.content.Context
import java.util.UUID

object PreferenceUtils {
    private const val PREFS_NAME = "com.vungn.camerax.PREFERENCE_FILE_KEY"
    private const val KEY_UUID = "device_uuid"
    private const val KEY_PIN = "user_pin"
    private const val KEY_FLAG = "first_time_flag"

    fun getUUID(context: Context): String {
        val sharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        var uuid = sharedPreferences.getString(KEY_UUID, null)
        if (uuid == null) {
            uuid = UUID.randomUUID().toString()
            sharedPreferences.edit().putString(KEY_UUID, uuid).apply()
        }
        return uuid
    }

    fun getPIN(context: Context): String? {
        val sharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return sharedPreferences.getString(KEY_PIN, null)
    }

    fun setPIN(context: Context, pin: String) {
        val sharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        sharedPreferences.edit().putString(KEY_PIN, pin).apply()
    }

    fun isFirstTime(context: Context): Boolean {
        val sharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return sharedPreferences.getBoolean(KEY_FLAG, true)
    }

    fun setFirstTime(context: Context, isFirstTime: Boolean) {
        val sharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        sharedPreferences.edit().putBoolean(KEY_FLAG, isFirstTime).apply()
    }
}
