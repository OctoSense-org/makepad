package dev.makepad.android;

import android.view.Surface;
import android.view.MotionEvent;

public class MakepadNative {
    // belongs to MakepadActivity class
    public native static void activityOnCreate(Object activity);
    public native static void activityOnStart();
    public native static void activityOnResume();
    public native static void activityOnPause();
    public native static void activityOnStop();
    public native static void activityOnDestroy();
    public native static void activityOnWindowFocusChanged(boolean has_focus);
    public static native void onAndroidParams(String cache_path, String data_path, float density, boolean isEmulator, String androidVersion, String buildNumber,
        String kernelVersion);

    public native static void initChoreographer(float deviceRefreshRate, int sdkVersion);

    public native static void onBackPressed();

    // belongs to QuadSurface class
    public native static void surfaceOnSurfaceCreated(Surface surface);
    public native static void surfaceOnSurfaceDestroyed(Surface surface);
    public static native void surfaceOnLongClick(float x, float y, int pointerId, long timeMillis);
    public static native void surfaceOnTouch(MotionEvent event);
    public native static void surfaceOnSurfaceChanged(Surface surface, int width, int height);
    public native static void surfaceOnKeyDown(int keycode, int meta_state, boolean is_repeat);
    public native static void surfaceOnKeyUp(int keycode, int meta_state);
    public native static void surfaceOnCharacter(int character);
    public native static void surfaceOnResizeTextIME(int keyboard_height, boolean is_open);
    public native static void surfaceOnPhysicalKeyboardChanged(boolean connected);
    public native static void surfaceOnSafeAreaInsets(float top, float right, float bottom, float left);

    // networking
    public native static void onHttpResponse(long id, long metadata_id, int status_code, String headers, byte[] body);
    public native static void onHttpRequestError(long id, long metadata_id, String error);
    public native static void onWebSocketMessage(byte[] message, long callback);
    public native static void onWebSocketClosed(long callback);
    public native static void onWebSocketError(String error, long callback);

    // clipboard
    public native static void onClipboardAction(String action);
    public native static void onClipboardPaste(String content);

    // selection handles
    public native static void onSelectionHandleDrag(int handle, int phase, float x, float y, long timeMillis);

    // IME events - unified text state notification (Java→Rust)
    // Called when IME changes text (composition, commit, delete, selection change)
    public native static void onImeTextStateChanged(
        String fullText,
        int selectionStart,
        int selectionEnd,
        int composingStart,
        int composingEnd
    );

    // IME editor action (for single-line inputs: Done, Go, Search, etc.)
    // Called when user presses the action button on the soft keyboard
    public native static void onImeEditorAction(int actionCode);

    // Native floating chat composer (an Android view floating over the GL
    // surface) submitted its text — user tapped send or pressed IME "Send".
    public native static void onComposerSubmit(String text);

    // Composer control buttons: open another app / switch to the next app.
    public native static void onComposerNewApp();
    public native static void onComposerSwitch();

    // The collapsed "+" FAB was tapped to unfold the composer (keeps the app's
    // composer_shown state in sync with the native overlay).
    public native static void onComposerExpand();

    // A runhtml web-card's JS called octos.invoke(tool, args) — bridged in from the
    // WebView's "octos_native" JavascriptInterface. Rust dispatches the tool and
    // resolves the card-side promise (callId) via evalSystemBrowserJs.
    public native static void onSystemBrowserInvoke(long browserId, long callId, String tool, String args);

    // The app was opened/resumed via a deep link (ACTION_VIEW URL) or a share
    // (ACTION_SEND text) — e.g. a YouTube link shared from another app.
    public native static void onDeepLink(String url);

    // Result of a native file picker (openFileDialog) — the picked file's name +
    // text contents, or a cancel/error. Resolves the card's octos.invoke("dialog.open").
    public native static void onDialogResult(long callId, String name, String content, boolean cancelled, String error);

    // Native streaming download (downloadFile): periodic progress, then completion.
    public native static void onDownloadProgress(long callId, long done, long total);
    public native static void onDownloadComplete(long callId, String path, String error);

    // A camera frame (NV21 luma plane) from the QR scanner overlay. Rust decodes
    // it; returns true if a QR was found (the caller then closes the scanner).
    public native static boolean onQrCameraFrame(byte[] luma, int width, int height);

    // midi
    public native static void onMidiDeviceOpened(String name, Object midi_device);
    
    // permissions
    public native static void onPermissionResult(String permission, int requestId, int status);

    // video playback
    public static native void onVideoPlaybackPrepared(long videoId, int videoWidth, int videoHeight, long duration, VideoPlayer surfaceTexture);
    public static native void onVideoPlaybackCompleted(long videoId);
    public static native void onVideoPlayerReleased(long videoId);
    public static native void onVideoDecodingError(long videoId, String error);
    public static native void onH264EncoderPacket(long encoderId, long ptsUs, int flags, byte[] data);
    public static native void onH264EncoderError(long encoderId, String error);
    public static native void onCameraPreviewSurfaceReady(long videoId, Surface surface, int width, int height);
    public static native void onCameraPreviewSurfaceDestroyed(long videoId);
}
