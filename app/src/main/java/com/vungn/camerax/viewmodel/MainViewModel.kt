package com.vungn.camerax.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class MainViewModel : ViewModel() {
    private val _isFirstRun = MutableStateFlow(true)
    val isFirstRun: StateFlow<Boolean> = _isFirstRun

    fun setFirstRun(isFirstRun: Boolean) {
        viewModelScope.launch {
            _isFirstRun.value = isFirstRun
        }
    }
}
