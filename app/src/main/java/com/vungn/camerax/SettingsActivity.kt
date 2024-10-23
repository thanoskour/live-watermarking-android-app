package com.vungn.camerax

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("SettingsActivity", "Starting SettingsActivity")

        setContentView(R.layout.settings_activity)

        Log.d("SettingsActivity", "Layout set")

        supportFragmentManager.beginTransaction()
            .replace(android.R.id.content, SettingsFragment())
            .commit()
    }
}
