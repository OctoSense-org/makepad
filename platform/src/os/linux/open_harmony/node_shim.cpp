// makepad's one ArkUI node, created natively (NDK) instead of by ArkTS.
//
// ArkTS hands over a NodeContent slot (a ContentSlot on the page); this file
// puts makepad's XComponent into it and listens to it natively: axis events
// (mouse wheel, two-finger touchpad), and the node's UI context, which the
// native input method needs to attach. Everything else about the XComponent
// (surface, touch, mouse, keys) is registered from Rust on the
// OH_NativeXComponent this returns.
//
// A shim rather than Rust bindings because ArkUI_NativeNodeAPI_1 is a table of
// function pointers whose order is the ABI: the SDK header checks it here.
// C++ because the SDK's ArkUI headers need a C++ translation unit.
#include <cstdint>

#include <ace/xcomponent/native_interface_xcomponent.h>
#include <arkui/native_interface.h>
#include <arkui/native_node.h>
#include <arkui/native_node_napi.h>
#include <arkui/native_type.h>
#include <arkui/ui_input_event.h>
#include <napi/native_api.h>

extern "C" void makepad_ohos_on_axis(double dx, double dy, float x, float y, int32_t tool);

static ArkUI_NativeNodeAPI_1 *g_api = nullptr;

static void on_node_event(ArkUI_NodeEvent *event) {
    if (OH_ArkUI_NodeEvent_GetEventType(event) != NODE_ON_AXIS) {
        return;
    }
    ArkUI_UIInputEvent *input = OH_ArkUI_NodeEvent_GetInputEvent(event);
    if (input == nullptr) {
        return;
    }
    makepad_ohos_on_axis(OH_ArkUI_AxisEvent_GetHorizontalAxisValue(input),
                         OH_ArkUI_AxisEvent_GetVerticalAxisValue(input),
                         OH_ArkUI_PointerEvent_GetX(input), OH_ArkUI_PointerEvent_GetY(input),
                         OH_ArkUI_UIInputEvent_GetToolType(input));
}

// 0 on success; the XComponent and the UI context come back through the out
// pointers. Must run on the ArkTS main thread (it is called from napi).
extern "C" int makepad_ohos_mount(napi_env env, napi_value content_value, OH_NativeXComponent **out_xcomponent,
                                  ArkUI_ContextHandle *out_context) {
    OH_ArkUI_GetModuleInterface(ARKUI_NATIVE_NODE, ArkUI_NativeNodeAPI_1, g_api);
    if (g_api == nullptr) {
        return -1;
    }
    ArkUI_NodeContentHandle content = nullptr;
    if (OH_ArkUI_GetNodeContentFromNapiValue(env, content_value, &content) != 0 || content == nullptr) {
        return -2;
    }
    ArkUI_NodeHandle node = g_api->createNode(ARKUI_NODE_XCOMPONENT);
    if (node == nullptr) {
        return -3;
    }
    ArkUI_AttributeItem id = {nullptr, 0, "makepad", nullptr};
    g_api->setAttribute(node, NODE_XCOMPONENT_ID, &id);
    ArkUI_NumberValue type[] = {{.i32 = ARKUI_XCOMPONENT_TYPE_SURFACE}};
    ArkUI_AttributeItem type_item = {type, 1, nullptr, nullptr};
    g_api->setAttribute(node, NODE_XCOMPONENT_TYPE, &type_item);
    ArkUI_NumberValue full[] = {{.f32 = 1.0f}};
    ArkUI_AttributeItem full_item = {full, 1, nullptr, nullptr};
    g_api->setAttribute(node, NODE_WIDTH_PERCENT, &full_item);
    g_api->setAttribute(node, NODE_HEIGHT_PERCENT, &full_item);
    ArkUI_NumberValue yes[] = {{.i32 = 1}};
    ArkUI_AttributeItem yes_item = {yes, 1, nullptr, nullptr};
    g_api->setAttribute(node, NODE_FOCUSABLE, &yes_item);
    g_api->setAttribute(node, NODE_DEFAULT_FOCUS, &yes_item);
    g_api->registerNodeEventReceiver(on_node_event);
    g_api->registerNodeEvent(node, NODE_ON_AXIS, 0, nullptr);
    if (OH_ArkUI_NodeContent_AddNode(content, node) != 0) {
        return -4;
    }
    *out_xcomponent = OH_NativeXComponent_GetNativeXComponent(node);
    *out_context = OH_ArkUI_GetContextByNode(node);
    return *out_xcomponent == nullptr ? -5 : 0;
}
