@file:OptIn(ExperimentalMaterial3Api::class)

package com.vungn.camerax

import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.os.Bundle
import android.provider.Settings
import android.text.InputFilter
import android.util.Log
import android.view.LayoutInflater
import android.widget.EditText
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.preference.PreferenceManager
import com.google.common.util.concurrent.ListenableFuture
import com.vungn.camerax.ui.CameraX
import com.vungn.camerax.ui.theme.CameraXTheme
import com.vungn.camerax.util.CameraXHelper
import java.util.UUID
import com.vungn.camerax.viewmodel.MainViewModel


class MainActivity : ComponentActivity() {
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private val cameraXHelper: CameraXHelper by lazy {
        CameraXHelper.getInstance()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val context = LocalContext.current
            val isProcessing by cameraXHelper.isProcessing.collectAsState()
            val lifecycleOwner = LocalLifecycleOwner.current
            val previewView = remember { PreviewView(context) }
            val isGranted = remember { mutableStateOf(false) }
            val snackBarHostState = remember { SnackbarHostState() }
            val useCase = cameraXHelper.useCase.collectAsState()
            val message = cameraXHelper.message.collectAsState()
            val cameraCapturing = cameraXHelper.cameraCapturing.collectAsState()
            val videoPause = cameraXHelper.videoPause.collectAsState()
            val torchState = cameraXHelper.torchState.collectAsState()
            val meteringPoint = cameraXHelper.meteringPoint.collectAsState()
            val aspectRatio = cameraXHelper.aspectRatio.collectAsState()
            val cameraLen = cameraXHelper.len.collectAsState()
            val barcodes = cameraXHelper.barcodes.collectAsState()
            val videoTimer = cameraXHelper.videoTimer.collectAsState()
            val filteredQualities = cameraXHelper.filteredQualities.collectAsState()
            val videoQuality = cameraXHelper.videoQuality.collectAsState()
            val rotation = cameraXHelper.rotation.collectAsState()
            val state = rememberTransformableState { zoomChange, _, _ ->
                cameraXHelper.camera?.cameraControl?.setZoomRatio(cameraXHelper.camera?.cameraInfo?.zoomState?.value?.zoomRatio!! * zoomChange)
            }
            val requestMultiplePermissions =
                rememberLauncherForActivityResult(contract = ActivityResultContracts.RequestMultiplePermissions(),
                    onResult = { permissions ->
                        Log.d(TAG, "is granted: ${permissions.all { true }}")
                        isGranted.value = permissions.all { true }
                    })

            val mainViewModel: MainViewModel = viewModel()

            val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(this)
            val isFirstRun = sharedPrefs.getBoolean("is_first_run", true)

            val isFirstRunState by mainViewModel.isFirstRun.collectAsState()

            if (isFirstRun && isFirstRunState) {
                mainViewModel.setFirstRun(false)
                showPINInputDialog(this) { pin ->
                    val uuid = getFormattedUUID(this)
                    with(sharedPrefs.edit()) {
                        putString("user_pin", pin)
                        putString("user_uuid", uuid)
                        putBoolean("is_first_run", true)
                        apply()
                    }
                }
            }



            LaunchedEffect(key1 = true, block = {
                requestMultiplePermissions.launch(
                    arrayOf(
                        Manifest.permission.CAMERA,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE,
                        Manifest.permission.RECORD_AUDIO
                    )
                )
            })


            LaunchedEffect(keys = arrayOf(
                isGranted.value,
                aspectRatio.value,
                cameraLen.value,
                useCase.value,
                videoQuality.value
            ), block = {
                if (isGranted.value) {
                    cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        cameraXHelper.bindPreview(
                            context = context,
                            lifecycleOwner = lifecycleOwner,
                            cameraProvider = cameraProvider,
                            previewView = previewView
                        )
                    }, ContextCompat.getMainExecutor(context))
                }
            })
            LaunchedEffect(key1 = message.value, block = {
                if (message.value != null) {
                    snackBarHostState.showSnackbar(message = message.value!!)
                }
            })


            CameraXTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background
                ) {

                    Scaffold(modifier = Modifier,
                        snackbarHost = { SnackbarHost(hostState = snackBarHostState) },
                    ) { padding ->
                        CameraX(
                            previewView = previewView,
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(padding)
                                .transformable(state = state),
                            useCase = useCase.value,
                            capturing = cameraCapturing.value,
                            videoPause = videoPause.value,
                            torchState = torchState.value,
                            meteringPoint = meteringPoint.value,
                            aspectRatio = aspectRatio.value,
                            cameraLen = cameraLen.value,
                            barcodes = barcodes.value,
                            videoTimer = videoTimer.value,
                            filteredQualities = filteredQualities.value,
                            videoQuality = videoQuality.value,
                            rotation = rotation.value,
                            imageCapture = cameraXHelper::imageCapturing,
                            changeTorchState = cameraXHelper::changeTorchState,
                            changeRatio = cameraXHelper::changeAspectRatio,
                            switchCamera = cameraXHelper::switchCamera,
                            videoCapture = cameraXHelper::videoCapturing,
                            stopVideoCapturing = cameraXHelper::stopVideoCapturing,
                            pauseVideoCapturing = cameraXHelper::pauseVideoCapturing,
                            resumeVideoCapturing = cameraXHelper::resumeVideoCapturing,
                            changeUseCase = cameraXHelper::changeUseCase,
                            changeVideoQuality = cameraXHelper::changeVideoQuality,

                        )
                        if (isProcessing) {
                            LoadingScreen("Processing image, please wait...")
                        }
                    }
                }
            }
        }
    }



    @Composable
    fun LoadingScreen(message: String = "Processing...") {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .fillMaxSize()
                .background(color = Color.Black.copy(alpha = 0.8F))
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator(color = Color.White)
                Spacer(modifier = Modifier.height(16.dp))
                Text(text = message, color = Color.White)
            }
        }
    }

    private fun showPINInputDialog(context: Context, onPINSet: (String) -> Unit) {
        val dialogView = LayoutInflater.from(context).inflate(R.layout.dialog_pin_input, null)
        val pinEditText = dialogView.findViewById<EditText>(R.id.pinEditText)

        // Set input filter to allow only 4 digits
        pinEditText.filters = arrayOf(InputFilter.LengthFilter(4))

        AlertDialog.Builder(context)
            .setTitle("Enter PIN")
            .setView(dialogView)
            .setPositiveButton("OK") { dialog, _ ->
                val pin = pinEditText.text.toString()
                if (pin.length == 4 && pin.all { it.isDigit() }) {
                    onPINSet(pin)
                } else {
                    Toast.makeText(context, "PIN must be exactly 4 digits", Toast.LENGTH_SHORT).show()
                }
                dialog.dismiss()
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
            }
            .create()
            .show()
    }




        private fun getFormattedUUID(context: Context): String {
            // Get the Android ID
            val androidId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
            Log.d("DeviceUtils", "Android ID: $androidId")

            // Concatenate the Android ID with itself
            val concatenatedId = androidId + androidId

            // Format it like a UUID: 8-4-4-4-12
            val formattedUuid = "${concatenatedId.substring(0, 8)}-${concatenatedId.substring(8, 12)}-${concatenatedId.substring(12, 16)}-${concatenatedId.substring(16, 20)}-${concatenatedId.substring(20)}"
            Log.d("DeviceUtils", "Formatted UUID-like ID: $formattedUuid")

            return formattedUuid
        }





    companion object {
        const val TAG = "MainActivity"
    }
}