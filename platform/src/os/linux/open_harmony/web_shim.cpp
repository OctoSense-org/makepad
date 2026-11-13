// makepad's system browsers on OpenHarmony: the parts of ArkWeb the NDK
// exposes, addressed by web tag.
//
// ArkTS has to declare each `Web` component (there is no web node in the NDK),
// and its controller is named with the browser's tag. From then on the page
// talks to Rust directly: `window.octos_native.invoke(callId, tool, args)` is a
// proxy registered here, and replies and events go back with runJavaScript.
//
// C++ for the same reason as node_shim.cpp: the ArkWeb APIs are structs of
// function pointers whose order is the ABI, checked here by the SDK header and
// by ARKWEB_MEMBER_EXISTS (a device may predate a member).
#include <cstddef>
#include <cstdint>

#include <web/arkweb_interface.h>
#include <web/arkweb_type.h>

// Rust (oh_web.rs): one `invoke(callId, tool, args)` from a page.
extern "C" void makepad_ohos_web_invoke(const char *tag, const char *call_id, size_t call_id_len, const char *tool,
                                        size_t tool_len, const char *args, size_t args_len);

// Main thread only (oh_web.rs runs every call there), so a plain global: a
// function-local static would need the C++ runtime's guard functions, which
// the app does not link.
static ArkWeb_ControllerAPI *g_controller = nullptr;

static ArkWeb_ControllerAPI *controller_api() {
    if (g_controller == nullptr) {
        g_controller = reinterpret_cast<ArkWeb_ControllerAPI *>(OH_ArkWeb_GetNativeAPI(ARKWEB_NATIVE_CONTROLLER));
    }
    return g_controller;
}

static void on_invoke(const char *tag, const ArkWeb_JavaScriptBridgeData *data, size_t size, void *) {
    if (size < 3 || data == nullptr) {
        return;
    }
    makepad_ohos_web_invoke(tag, reinterpret_cast<const char *>(data[0].buffer), data[0].size,
                            reinterpret_cast<const char *>(data[1].buffer), data[1].size,
                            reinterpret_cast<const char *>(data[2].buffer), data[2].size);
}

static const ArkWeb_ProxyMethod kMethods[] = {{"invoke", on_invoke, nullptr}};
static const ArkWeb_ProxyObject kBridge = {"octos_native", kMethods, 1};

// Expose `octos_native` to the documents `tag` loads from now on. Call once
// its controller is attached and before loading the first document. 0 = ok.
extern "C" int makepad_arkweb_register_bridge(const char *tag) {
    ArkWeb_ControllerAPI *api = controller_api();
    if (api == nullptr || ARKWEB_MEMBER_MISSING(api, registerJavaScriptProxy)) {
        return -1;
    }
    api->registerJavaScriptProxy(tag, &kBridge);
    return 0;
}

static void on_js_result(const char *, const ArkWeb_JavaScriptBridgeData *, void *) {}

// Evaluate `js` (len bytes, not NUL-terminated) in the document `tag` shows.
extern "C" int makepad_arkweb_run_js(const char *tag, const char *js, size_t len) {
    ArkWeb_ControllerAPI *api = controller_api();
    if (api == nullptr || ARKWEB_MEMBER_MISSING(api, runJavaScript)) {
        return -1;
    }
    ArkWeb_JavaScriptObject object = {reinterpret_cast<const uint8_t *>(js), len, on_js_result, nullptr};
    api->runJavaScript(tag, &object);
    return 0;
}
