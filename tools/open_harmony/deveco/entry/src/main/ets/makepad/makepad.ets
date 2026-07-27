import display from '@ohos.display';
import deviceInfo from '@ohos.deviceInfo';
import inputMethod from '@ohos.inputMethod';
import { window } from '@kit.ArkUI';
import common from '@ohos.app.ability.common';
import resourceManager from '@ohos.resourceManager';
import webview from '@ohos.web.webview';
import picker from '@ohos.file.picker';
import fileIo from '@ohos.file.fs';
import util from '@ohos.util';

declare type Any = ESObject;

/// The observable half of a webview card: geometry only.
///
/// Deliberately holds NO WebviewController. An @Observed object handed to
/// @State/@ObjectLink is wrapped in a proxy, and calling a native-backed method
/// like loadData() through that proxy silently does nothing — no exception, no
/// navigation, card stays blank. Controllers therefore live in a plain Map in
/// ArkGlue (see WebViewCtl) and are never routed through observed state.
@Observed
export class WebViewSpec {
  id: string = '';
  src: string = 'about:blank';
  x: number = 0;
  y: number = 0;
  w: number = 1;
  h: number = 1;
  visible: boolean = false;

  constructor(id: string) {
    this.id = id;
  }
}

/// The non-observable half: the real controller plus anything Rust sent before
/// the ArkUI Web component existed to receive it.
export class WebViewCtl {
  controller: webview.WebviewController = new webview.WebviewController();
  bridge: OctosNativeBridge | undefined = undefined;
  attached: boolean = false;
  pendingUrl: string = '';
  pendingHtml: string = '';
  pendingBaseUrl: string = '';
  pendingJs: string[] = [];

  loadUrl(url: string): void {
    if (!this.attached) {
      this.pendingUrl = url;
      this.pendingHtml = '';
      return;
    }
    try {
      this.controller.loadUrl(url);
    } catch (e) {
      console.error(`makepad webview loadUrl failed: ${JSON.stringify(e)}`);
    }
  }

  loadHtml(html: string, baseUrl: string): void {
    if (!this.attached) {
      this.pendingHtml = html;
      this.pendingBaseUrl = baseUrl;
      this.pendingUrl = '';
      return;
    }
    // Do NOT pass `undefined` for the optional trailing arguments: ArkWeb
    // type-checks them and rejects the whole call with code 401, so the
    // document silently never loads. Pass only as many arguments as are real.
    try {
      if (baseUrl.length > 0) {
        this.controller.loadData(html, 'text/html', 'UTF-8', baseUrl);
      } else {
        this.controller.loadData(html, 'text/html', 'UTF-8');
      }
    } catch (e) {
      console.error(`makepad webview loadData failed: ${JSON.stringify(e)}`);
      try {
        this.controller.loadUrl(
          'data:text/html;charset=utf-8,' + encodeURIComponent(html));
      } catch (e2) {
        console.error(`makepad webview data-url fallback failed: ${JSON.stringify(e2)}`);
      }
    }
  }

  evalJs(js: string): void {
    if (!this.attached) {
      this.pendingJs.push(js);
      return;
    }
    try {
      this.controller.runJavaScript(js);
    } catch (e) {
      console.error(`makepad webview runJavaScript failed: ${JSON.stringify(e)}`);
    }
  }

  historyGo(delta: number): void {
    if (!this.attached) {
      return;
    }
    try {
      this.controller.backOrForward(delta);
    } catch (e) {
      console.error(`makepad webview backOrForward failed: ${JSON.stringify(e)}`);
    }
  }

  /// Called once the ArkUI Web component has bound this controller.
  flushPending(): void {
    this.attached = true;
    console.info(`MKPWEB flushPending html=${this.pendingHtml.length} url=${this.pendingUrl}`);
    if (this.pendingHtml.length > 0) {
      const html = this.pendingHtml;
      const base = this.pendingBaseUrl;
      this.pendingHtml = '';
      this.loadHtml(html, base);
    } else if (this.pendingUrl.length > 0) {
      const url = this.pendingUrl;
      this.pendingUrl = '';
      this.loadUrl(url);
    }
    const js = this.pendingJs;
    this.pendingJs = [];
    for (const one of js) {
      this.evalJs(one);
    }
  }
}

export interface MakepadXComponentInterface{
  onCreate(ark_ts: object):void;
  handleInsertTextEvent(text:String):void;
  handleDeleteLeftEvent(length: number):void;
  handleKeyboardStatus(isOpen:boolean, keyboardHeight:number):void;
  handleComposerSubmit(text: string):void;
  handleComposerNewApp():void;
  handleComposerSwitch():void;
  handleComposerExpand():void;
  handleSystemBrowserInvoke(browserId: string, callId: string, tool: string, args: string):void;
  handleDialogResult(callId: string, name: string, content: string, cancelled: boolean, error: string):void;
}

/// The `octos_native` object injected into a webview card's JS context, the
/// OHOS counterpart of Android's `addJavascriptInterface(..., "octos_native")`.
///
/// A card calls `octos_native.invoke(callId, tool, argsJson)`; the WebCard
/// widget on the Rust side dispatches the tool (fs.*, dialog.open, share,
/// clipboard.write, notify, http.fetch, download — i.e. real native components)
/// and resolves the card's promise by evaluating JS back into the page.
///
/// ids are stringified on the way out: they are u64/i64 and JS numbers are f64,
/// so passing them as numbers loses precision past 2^53.
export class OctosNativeBridge {
  private browserId: string = '';

  constructor(browserId: string) {
    this.browserId = browserId;
  }

  invoke(callId: string, tool: string, args: string): void {
    ArkGlue.getInstance().webviewInvoke(this.browserId, String(callId), tool, args);
  }
}

/// State of the native chat composer overlay.
///
/// The composer is NATIVE here for the same reasons it is native on Android: a
/// full-screen card's PortalList swallows taps aimed at a makepad-drawn
/// composer, and makepad has no text-input bridge on OHOS, so a makepad
/// TextInput could never receive characters at all.
@Observed
export class ComposerState {
  visible: boolean = false;
  expanded: boolean = false;
}


function getDensity(): number {
  try {
    let displayClass = display.getDefaultDisplaySync();
    console.info('Test densityDPI:' + JSON.stringify(displayClass.densityDPI));
    return displayClass.densityDPI / 160;
  } catch (exception) {
    console.error('Failed to obtain the default display object. Code: ' + JSON.stringify(exception));
    return 3;
  }
}

function getDeviceType(): string {
  let device_type: string = deviceInfo.deviceType;
  if (device_type == "") {
    console.error("deviceInfo.deviceType is empty string!")
  } else {
    console.info("Device type is " + device_type)
  }
  return device_type;
}

export class ArkGlue{
  private static instance : ArkGlue;
  static initInstance(context: common.Context) : ArkGlue {
    if(ArkGlue.instance == null) {
      ArkGlue.instance = new ArkGlue(context);
    }
      return ArkGlue.instance;
  }

  static getInstance() : ArkGlue {
    return ArkGlue.instance;
  }

  context: common.Context;

  deviceType = getDeviceType();
  osFullName = deviceInfo.osFullName;
  displayDensity = getDensity();
  resMgr : resourceManager.ResourceManager;
  filesDir : string;
  cacheDir : string;
  tempDir : string;

  keyboardHeight : number = 1020;
  xComponentContext : MakepadXComponentInterface | undefined = undefined;

  private inputMethodController : inputMethod.InputMethodController;
  private textConfig: inputMethod.TextConfig;
  private imcFlag = false;

  constructor(context: common.Context) {
    this.context = context;
    this.resMgr = this.context.resourceManager;
    this.filesDir = this.context.filesDir;
    this.cacheDir = this.context.cacheDir;
    this.tempDir = this.context.tempDir;
    this.inputMethodController = inputMethod.getController();
    this.textConfig = {
      inputAttribute: {
        textInputType: 0,
        enterKeyType: 1
      }
    };
  }

  // ---- system browser (webview cards), called from Rust ----
  //
  // Geometry arrives in vp, which is what ArkUI lays out in, so it is used
  // verbatim. Every method is tolerant of an unknown id: Rust does NOT
  // guarantee spawn arrives first (a card already `spawned` on the Rust side
  // re-sends only set_html after a widget rebuild), so ops create on demand.

  webviews: WebViewSpec[] = [];
  private ctls: Map<string, WebViewCtl> = new Map<string, WebViewCtl>();
  onWebViewsChanged: (specs: WebViewSpec[]) => void = () => {};

  /// The raw controller for an id, never routed through observed state.
  controllerFor(id: string): webview.WebviewController {
    return this.ctlEnsure(id).controller;
  }

  /// The `octos_native` proxy object for a card's JS context.
  bridgeFor(id: string): OctosNativeBridge {
    const ctl = this.ctlEnsure(id);
    if (ctl.bridge === undefined) {
      ctl.bridge = new OctosNativeBridge(id);
    }
    return ctl.bridge;
  }

  /// A card called `octos_native.invoke(...)` — hand it to Rust.
  webviewInvoke(id: string, callId: string, tool: string, args: string): void {
    this.xComponentContext?.handleSystemBrowserInvoke(id, callId, tool, args);
  }

  /// Open the SYSTEM file picker for a card's `octos.invoke("dialog.open")`.
  ///
  /// This is the real OS component (DocumentViewPicker), the OHOS counterpart
  /// of Android's ACTION_OPEN_DOCUMENT intent — not an in-page file input. The
  /// picked file is read here and handed back as text, matching the Android
  /// contract (name + content + cancelled + error).
  openFileDialog(callId: string, mime: string): void {
    this.runFilePicker(callId, mime);
  }

  private async runFilePicker(callId: string, mime: string): Promise<void> {
    try {
      const options = new picker.DocumentSelectOptions();
      options.maxSelectNumber = 1;
      const uris: string[] = await new picker.DocumentViewPicker(this.context).select(options);
      if (!uris || uris.length === 0) {
        this.xComponentContext?.handleDialogResult(callId, '', '', true, '');
        return;
      }
      const uri = uris[0];
      const file = fileIo.openSync(uri, fileIo.OpenMode.READ_ONLY);
      try {
        const stat = fileIo.statSync(file.fd);
        const buf = new ArrayBuffer(stat.size);
        fileIo.readSync(file.fd, buf);
        const content = util.TextDecoder.create('utf-8').decodeToString(new Uint8Array(buf));
        const name = uri.substring(uri.lastIndexOf('/') + 1);
        this.xComponentContext?.handleDialogResult(callId, name, content, false, '');
      } finally {
        fileIo.closeSync(file);
      }
    } catch (e) {
      // A user-cancelled picker also throws on some releases; report it as an
      // error rather than a silent cancel so the card's promise always settles.
      this.xComponentContext?.handleDialogResult(callId, '', '', false, `picker: ${JSON.stringify(e)}`);
    }
  }

  webviewAttached(id: string): void {
    this.ctlEnsure(id).flushPending();
  }

  private ctlEnsure(id: string): WebViewCtl {
    let ctl = this.ctls.get(id);
    if (ctl === undefined) {
      ctl = new WebViewCtl();
      this.ctls.set(id, ctl);
    }
    return ctl;
  }

  private webviewEnsure(id: string): WebViewSpec {
    for (const spec of this.webviews) {
      if (spec.id === id) {
        return spec;
      }
    }
    const spec = new WebViewSpec(id);
    this.webviews.push(spec);
    this.webviewNotify();
    return spec;
  }

  private webviewNotify(): void {
    // Hand out a fresh array so @State in the page sees a new reference. Only
    // called on structural changes (add/remove) — see webviewUpdate.
    this.onWebViewsChanged(this.webviews.slice());
  }

  webviewSpawn(id: string, url: string): void {
    const spec = this.webviewEnsure(id);
    if (url.length > 0) {
      spec.src = url;
    }
    this.ctlEnsure(id);
    console.info(`MKPWEB spawn id=${id} url=${url}`);
  }

  webviewUpdate(id: string, x: number, y: number, w: number, h: number, visible: boolean): void {
    const spec = this.webviewEnsure(id);
    if (spec.x !== x || spec.y !== y || spec.w !== w || spec.h !== h || spec.visible !== visible) {
      console.info(`MKPWEB rect id=${id} x=${x} y=${y} w=${w} h=${h} vis=${visible}`);
    }
    spec.x = x;
    spec.y = y;
    // A zero-sized Web component is invalid; keep it at 1vp while hidden.
    spec.w = w > 0 ? w : 1;
    spec.h = h > 0 ? h : 1;
    spec.visible = visible;
    // Deliberately NOT webviewNotify(): this runs once per frame, and handing
    // @State a new array that often rebuilds the Web component and wipes the
    // loaded document. The page observes these property writes via @ObjectLink.
  }

  webviewSetUrl(id: string, url: string, replace: boolean): void {
    this.webviewEnsure(id);
    this.ctlEnsure(id).loadUrl(url);
  }

  webviewSetHtml(id: string, html: string, baseUrl: string): void {
    this.webviewEnsure(id);
    console.info(`MKPWEB setHtml id=${id} len=${html.length} base=${baseUrl}`);
    this.ctlEnsure(id).loadHtml(html, baseUrl);
  }

  webviewEvalJs(id: string, js: string): void {
    this.webviewEnsure(id);
    this.ctlEnsure(id).evalJs(js);
  }

  webviewHistoryGo(id: string, delta: number): void {
    this.webviewEnsure(id);
    this.ctlEnsure(id).historyGo(delta);
  }

  /// Detach hides the browser but keeps its document alive, so a card that
  /// scrolls out of view and back does not have to reload.
  webviewDetach(id: string): void {
    for (const spec of this.webviews) {
      if (spec.id === id) {
        spec.visible = false;
      }
    }
  }

  webviewClose(id: string): void {
    this.webviews = this.webviews.filter((spec: WebViewSpec) => spec.id !== id);
    this.ctls.delete(id);
    this.webviewNotify();
  }

  // ---- native composer overlay ----
  //
  // Rust drives visibility through the four Show/Hide/Expand/Collapse ops; the
  // overlay reports back through the XComponent's napi callbacks.

  composer: ComposerState = new ComposerState();

  composerShow(): void {
    this.composer.visible = true;
  }

  composerHide(): void {
    this.composer.visible = false;
    this.composer.expanded = false;
  }

  composerExpand(): void {
    this.composer.visible = true;
    this.composer.expanded = true;
  }

  composerCollapse(): void {
    this.composer.expanded = false;
  }

  /// The collapsed FAB was tapped: unfold locally AND tell the app, so its
  /// `composer_shown` matches and a later sync does not re-fold the pill the
  /// user just opened.
  composerUnfoldTapped(): void {
    this.composer.expanded = true;
    this.xComponentContext?.handleComposerExpand();
  }

  composerSubmitText(text: string): void {
    if (text.length === 0) {
      return;
    }
    this.xComponentContext?.handleComposerSubmit(text);
  }

  composerNewAppTapped(): void {
    this.xComponentContext?.handleComposerNewApp();
  }

  composerSwitchTapped(): void {
    this.xComponentContext?.handleComposerSwitch();
  }

  showKeyBoard(): void {
    this.showTextInput();
  }

  hideKeyBoard(): void {
    this.hideTextInput();
  }

  private async showTextInput(): Promise<void> {
    this.listenKeyboardHeightChange();
    await this.attach(true);
      if (this.imcFlag != true) {
      this.listenKeyBoardEvent();
    }
    this.inputMethodController.showTextInput().then(() => {
      console.debug("Succeeded in showing softKeyboard");
    }).catch((err: Any) => {
      console.error("Failed to show softKeyboard:" + JSON.stringify(err));
    });
  }

  private async hideTextInput(): Promise<void> {
    this.inputMethodController.hideTextInput().then(() => {
      console.debug("Succeeded in hide softKeyboard");
    }).catch((err: Any) => {
      console.error("Failed to hide softKeyboard:" + JSON.stringify(err));
    })
    this.cancelKeyboardHeightChange();
  }

  private async attach(showKeyboard: boolean): Promise<void> {
    try {
      await this.inputMethodController.attach(showKeyboard, this.textConfig);
    } catch (err) {
      console.error("Failed to attach:" + JSON.stringify(err));
    }
  }

  private listenKeyBoardEvent(): void {
    try {
      this.inputMethodController.on('insertText', (text) => {
      this.xComponentContext?.handleInsertTextEvent(text);
      console.debug("insertText: " + text);
    });
    } catch (err) {
      console.error("Failed to subscribe insertText:" + JSON.stringify(err));
      this.cancelListenKeyBoardEvent();
      return;
    }

    try {
      this.inputMethodController.on('deleteLeft', (length) => {
        this.xComponentContext?.handleDeleteLeftEvent(length);
        console.debug("deleteLeft, length:" + JSON.stringify(length));
      })
    } catch (err) {
      console.error("Failed to subscribe deleteLeft:" + JSON.stringify(err));
      this.cancelListenKeyBoardEvent();
      return;
    }

    try {
      this.inputMethodController.on('sendKeyboardStatus', (state) => {
        if (state == inputMethod.KeyboardStatus.HIDE) {
          this.xComponentContext?.handleKeyboardStatus(false, 0);
          this.inputMethodController.detach();
          this.cancelListenKeyBoardEvent();
        }
        console.debug("sendKeyboardStatus, state:" + JSON.stringify(state));
      })
    } catch (err) {
      console.error("Failed to subscribe sendKeyboardStatus:" + JSON.stringify(err));
      this.cancelListenKeyBoardEvent();
      return;
    }

    console.debug("listenKeyBoardEvent success");
    this.imcFlag = true;
  }

  private cancelListenKeyBoardEvent(): void {
    this.inputMethodController.off('insertText');
    this.inputMethodController.off('deleteLeft');
    this.inputMethodController.off('sendKeyboardStatus');
    this.imcFlag = false;
  }

  private listenKeyboardHeightChange() {
    window.getLastWindow(getContext(this)).then(currentWindow => {
      try {
        currentWindow.on('keyboardHeightChange', (data) => {
          console.info('Succeeded in enabling the listener for keyboard height changes. Data: ' + JSON.stringify(data));
          if (data != 0) {
            this.xComponentContext?.handleKeyboardStatus(true,data);
          }
        });
      } catch (exception) {
        console.error(`Failed to enable the listener for keyboard height changes. Cause code: ${exception.code}, message: ${exception.message}`);
      }
    })
  }

  private cancelKeyboardHeightChange() {
    window.getLastWindow(getContext(this)).then(currentWindow => {
      try {
        currentWindow.off('keyboardHeightChange');
      } catch (exception) {
        console.error(`Failed to enable the listener for keyboard height changes. Cause code: ${exception.code}, message: ${exception.message}`);
      }
    })
  }

}

export interface XComponentAttrs {
  id: string;
  type: number;
  libraryname: string;
}